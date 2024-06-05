import argparse
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from os.path import abspath
from pathlib import Path

import numpy as np
import requests
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from torch import cat, cuda
from torch import device as torch_device
from torch import load, long, no_grad, optim
from torch import random as torch_random
from torch import save
from torch.nn import BatchNorm2d, CrossEntropyLoss, Linear, init
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader.data_list_loader import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from tqdm import tqdm

from src.dag_gnn import ToCanonizedDirectedAcyclicGraph
from task_graph.tudataset.dagmlp_model import DAGMLP
from utils import PrinterLogger


def reset_to_kaiming_uniform(model):
    for m in model.modules():
        if isinstance(m, Linear):
            init.normal_(m.weight, 0, init.calculate_gain("relu") / np.sqrt(m.weight.size(1)))
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            m.reset_running_stats()
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


class Patience:
    """
    Implement common "patience" technique
    """

    def __init__(self, patience=20, use_loss=True, save_path=None):
        self.local_val_optimum = float("inf") if use_loss else -float("inf")
        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1
        self.val_loss, self.val_acc = None, None
        self.save_path = save_path

    def stop(self, epoch, val_loss, val_acc=None, model=None):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]):
                    save({"epoch": epoch + 1, "state_dict": model.state_dict()}, self.save_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc >= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_acc
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]):
                    save({"epoch": epoch + 1, "state_dict": model.state_dict()}, self.save_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience


def train(model, data, y, criterion, optimizer, scheduler):
    output = model(data)
    loss_train = criterion(output, y.to(output.device))
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    scheduler.step()
    return output, loss_train


@no_grad()
def eval(model, data, y, criterion):
    output = model(data)
    loss_test = criterion(output, y.to(output.device))
    return output, loss_test


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, use_multi_gpu):
    train_loss = 0
    train_correct = 0
    model.train()
    for idx, data in enumerate(train_loader):
        if not use_multi_gpu:
            data = data.to(device)
        y = cat([d.y for d in data], dim=0) if use_multi_gpu else data.y
        output, loss = train(model, data, y, criterion, optimizer, scheduler)
        num_graphs = len(data) if use_multi_gpu else data.num_graphs
        train_loss += loss.item() * num_graphs
        prediction = output.max(1)[1].type_as(y)
        train_correct += prediction.eq(y.double()).sum().item()

    train_acc = train_correct / len(train_loader.dataset)
    train_loss = train_loss / len(train_loader.dataset)

    val_loss = 0
    val_correct = 0
    model.eval()
    for idx, data in enumerate(val_loader):
        if not use_multi_gpu:
            data = data.to(device)
        y = cat([d.y for d in data], dim=0) if use_multi_gpu else data.y
        num_graphs = len(data) if use_multi_gpu else data.num_graphs
        output, loss = eval(model, data, y, criterion)
        val_loss += loss.item() * num_graphs
        prediction = output.max(1)[1].type_as(y)
        val_correct += prediction.eq(y.double()).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)

    return train_acc * 100, train_loss, val_acc * 100, val_loss


def validate_batch_size(length, batch_size):
    return length % batch_size == 1


def get_train_val_test_loaders(root, dataset_name, train_index, val_index, test_index, num_layers, k, use_multi_gpu):
    canonized_dataset_path = os.path.join(root, dataset_name, f"canonized_num_layers_{num_layers}_k_{k}.pkl")
    dataset_list = load(canonized_dataset_path)

    train_set = [dataset_list[i] for i in train_index]
    val_set = [dataset_list[i] for i in val_index]
    test_set = [dataset_list[i] for i in test_index]

    if use_multi_gpu:
        train_loader = DataListLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=validate_batch_size(len(train_set), args.batch_size),
        )
        val_loader = DataListLoader(val_set, batch_size=args.batch_size, num_workers=0, shuffle=False)
        test_loader = DataListLoader(test_set, batch_size=args.batch_size, num_workers=0, shuffle=False)
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=validate_batch_size(len(train_set), args.batch_size),
        )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def read_json_from_url(dataset_name):
    url_dict = {
        "DD": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/DD_splits.json",
        "ENZYMES": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/ENZYMES_splits.json",
        "NCI1": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/NCI1_splits.json",
        "PROTEINS": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/PROTEINS_full_splits.json",
        "IMDB-BINARY": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/IMDB-BINARY_splits.json",
        "IMDB-MULTI": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/IMDB-MULTI_splits.json",
        "COLLAB": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/COLLAB_splits.json",
        "REDDIT-BINARY": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/REDDIT-BINARY_splits.json",
        "REDDIT-MULTI-5K": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/REDDIT-MULTI-5K_splits.json",
    }
    try:
        response = requests.get(url_dict[dataset_name])
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return json.loads(response.text)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    except KeyError:
        print(f"No splits available for {dataset_name}")
        return None


def compute_max_degree(dataset):
    max_degree = 0
    degrees = []
    for data in dataset:
        degrees += [degree(data.edge_index[0], dtype=long)]
        max_degree = max(max_degree, degrees[-1].max().item())
    return max_degree


def main(args):
    device = torch_device("cuda") if all([args.device == "cuda", cuda.is_available()]) else torch_device("cpu")

    root = Path(__file__).resolve().parent.parent.parent.joinpath("data")
    number_of_layers = args.number_of_layers
    dataset_args = {
        "use_node_attr": False,
        "transform": None,
    }

    naive_dataset = TUDataset(root=root, name=args.dataset_name, **dataset_args)
    # Update `dataset_args` based on `dataset_name`
    if "ENZYMES" in args.dataset_name:
        dataset_args.update(use_node_attr=True)
    elif "PROTEINS" in args.dataset_name:
        dataset_args.update(use_node_attr=True)
    elif "IMDB-BINARY" in args.dataset_name:
        max_degree = compute_max_degree(naive_dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "IMDB-MULTI" in args.dataset_name:
        max_degree = compute_max_degree(naive_dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "COLLAB" in args.dataset_name:
        max_degree = compute_max_degree(naive_dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "REDDIT-BINARY" in args.dataset_name:
        max_degree = compute_max_degree(naive_dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "REDDIT-MULTI-5K" in args.dataset_name:
        max_degree = compute_max_degree(naive_dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))

    naive_dataset = TUDataset(root=root, name=args.dataset_name, **dataset_args)
    # BUILD DATASET FOR NUMBERS OF LAYERS
    for num_layers in number_of_layers:
        canonized_dataset_path = os.path.join(
            root, args.dataset_name, f"canonized_num_layers_{num_layers}_k_{args.k}.pkl"
        )
        if not os.path.exists(canonized_dataset_path):
            os.makedirs(os.path.join(root, args.dataset_name), exist_ok=True)
            print(f"Building canonized dataset for {args.dataset_name} with {num_layers} layers and {args.k} k")
            dag_transform = ToCanonizedDirectedAcyclicGraph(None, num_layers, args.k, 1, False)
            start_time = time.time()
            data_list = Parallel(n_jobs=-1)(delayed(dag_transform)(d) for d in tqdm(naive_dataset))
            end_time = time.time()
            runtime_path = os.path.join(root, args.dataset_name, f"runtime_num_layers_{num_layers}_k_{args.k}.txt")
            with open(runtime_path, "w") as f:
                f.write(str(end_time - start_time))
            save(data_list, canonized_dataset_path)
            print(
                f"Canonized dataset for {args.dataset_name} "
                f"with {num_layers} layers and k={args.k} saved in {canonized_dataset_path}"
            )
        else:
            print(f"Canonized dataset for {args.dataset_name} "
                  f"with {num_layers} layers and k={args.k} already exists")

    if args.build_only:
        print("Exiting...")
        exit(0)

    y = [d.y.item() for d in naive_dataset]
    features_dim = naive_dataset[0].x.shape[1]
    n_classes = len(np.unique(y))
    criterion = CrossEntropyLoss()
    splits = read_json_from_url(args.dataset_name)
    if splits is None:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        splits = [
            {
                "model_selection": [{"train": train_index.tolist(), "validation": test_index.tolist()}],
                "test": test_index.tolist(),
            }
            for it, (train_index, test_index) in enumerate(skf.split(np.zeros(len(y)), y))
        ]

    # LOGGING AND DIRECTORY
    if not os.path.exists(os.path.join("results", args.dataset_name)):
        os.makedirs(os.path.join("results", args.dataset_name))
    if not os.path.exists(os.path.join("logs", args.dataset_name)):
        os.makedirs(os.path.join("logs", args.dataset_name))
    if not os.path.exists(os.path.join("models", args.dataset_name)):
        os.makedirs(os.path.join("models", args.dataset_name))

    now = "-".join(str(datetime.datetime.today()).split()).split(".")[0].replace(":", ".")
    logger_name = f"layers_{''.join(map(str, number_of_layers))}-k_{args.k}-bs_{args.batch_size}_{now}"

    if args.tensorboard:
        tensorboard_path = abspath(os.path.join("tensorboard", args.dataset_name, logger_name))
        print("Logging with tensorboard in path {}".format(tensorboard_path))
        writer = SummaryWriter(tensorboard_path)

    logging.basicConfig(
        filename=os.path.join("logs", args.dataset_name, logger_name + ".log"),
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        filemode="w",
    )
    log = PrinterLogger(logging.getLogger(__name__))
    log.print_and_log(args.__dict__)

    all_accuracies_folds = []
    mean_accuracies_folds = []
    best_config_across_folds = []

    np.random.seed(10)
    torch_random.manual_seed(10)
    for it in range(10):
        log.info("-" * 30 + f"ITERATION {str(it + 1)}" + "-" * 30)
        print("-" * 30 + f"ITERATION {str(it + 1)}" + "-" * 30)
        train_index = splits[it]["model_selection"][0]["train"]
        val_index = splits[it]["model_selection"][0]["validation"]
        test_index = splits[it]["test"]

        loop_counter = 1
        result_dict = defaultdict(list)
        best_acc_across_folds = -float(np.inf)
        best_loss_across_folds = float(np.inf)
        n_params = len(number_of_layers) * len(args.dim_embeddings) * len(args.lrs)
        model_selection_epochs = args.epochs
        if n_params == 1:
            log.print_and_log(f"Only one configuration to search for, skipping model selection (train for one epoch)")
            model_selection_epochs = 1
        for lr in args.lrs:
            for num_layers in number_of_layers:
                for dim_embedding in args.dim_embeddings:

                    ################################
                    #       MODEL SELECTION       #
                    ###############################
                    train_loader, val_loader, _ = get_train_val_test_loaders(
                        root,
                        args.dataset_name,
                        train_index,
                        val_index,
                        test_index,
                        num_layers,
                        args.k,
                        args.use_multi_gpu,
                    )

                    early_stopper = Patience(patience=args.patience, use_loss=False)

                    params = {"dim_embedding": dim_embedding, "num_layers": num_layers, "lr": lr}

                    model_config = {
                        "dim_embedding": params["dim_embedding"],
                        "num_layers": params["num_layers"],
                        "dim_features": features_dim,
                        "dim_target": n_classes,
                        "dropout": 0.5,
                        "train_eps": True,
                        "aggregation": "mean",
                        "combine_multi_height": args.combine_multi_height,
                        "use_linear_after_readout": args.use_linear_after_readout,
                    }

                    best_val_loss, best_val_acc = float("inf"), 0
                    epoch, val_loss, val_acc = 0, float("inf"), 0

                    try:
                        model = DAGMLP(**model_config)
                        reset_to_kaiming_uniform(model)

                        if args.use_multi_gpu:
                            number_of_available_gpus = cuda.device_count()
                            model = DataParallel(model, device_ids=[*range(number_of_available_gpus)])

                        model = model.to(device)

                        log.print_and_log(
                            f"Model # Parameters {sum([p.numel() for p in model.parameters()])}, {params}"
                        )
                        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

                        pbar_train = tqdm(range(model_selection_epochs), desc="Epoch 0 Loss 0")
                        for epoch in pbar_train:
                            train_acc, train_loss, val_acc, val_loss = train_and_validate_model(
                                model,
                                train_loader,
                                val_loader,
                                criterion,
                                optimizer,
                                scheduler,
                                device,
                                args.use_multi_gpu,
                            )
                            if args.tensorboard:
                                writer.add_scalar(
                                    f"Fold {it + 1} - "
                                    f"(A) Model Selection/config-{loop_counter}/Train Loss",
                                    train_loss,
                                    epoch,
                                )
                                writer.add_scalar(
                                    f"Fold {it + 1} - "
                                    f"(A) Model Selection/config-{loop_counter}/Train Accuracy",
                                    train_acc,
                                    epoch,
                                )
                                writer.add_scalar(
                                    f"Fold {it + 1} - "
                                    f"(A) Model Selection/config-{loop_counter}/Val Loss",
                                    val_loss,
                                    epoch,
                                )
                                writer.add_scalar(
                                    f"Fold {it + 1} - "
                                    f"(A) Model Selection/config-{loop_counter}/Val Accuracy",
                                    val_acc,
                                    epoch,
                                )

                            if early_stopper.stop(epoch, val_loss, val_acc):
                                break

                            best_acc_across_folds = (
                                early_stopper.val_acc
                                if early_stopper.val_acc > best_acc_across_folds
                                else best_acc_across_folds
                            )
                            best_loss_across_folds = (
                                early_stopper.val_loss
                                if early_stopper.val_loss < best_loss_across_folds
                                else best_loss_across_folds
                            )

                            pbar_train.set_description(
                                f"MS {loop_counter}/{n_params} "
                                f"Epoch {epoch + 1} "
                                f"Val loss {val_loss:0.2f} "
                                f"Val acc {val_acc:0.1f} "
                                f"Best Val Loss {early_stopper.val_loss:0.2f} "
                                f"Best Val Acc {early_stopper.val_acc:0.1f} "
                                f"Best Fold Val Acc  {best_acc_across_folds:0.1f} "
                                f"Best Fold Val Loss {best_loss_across_folds:0.2f}"
                            )

                        best_val_loss, best_val_acc = early_stopper.val_loss, early_stopper.val_acc
                    except Exception as e:
                        print(f"Error during evaluating params {params} | {e}")
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad
                        cuda.empty_cache()

                    result_dict["config"].append(params)
                    result_dict["best_val_acc"].append(best_val_acc)
                    result_dict["best_val_loss"].append(best_val_loss)
                    log.info(
                        f"MS {loop_counter}/{n_params} "
                        f"Epoch {epoch + 1} "
                        f"Best Epoch {early_stopper.best_epoch} "
                        f"Val acc {val_acc:0.1f} "
                        f"Best Val Acc {best_val_acc:0.2f} "
                        f"Best Fold Val Acc  {best_acc_across_folds:0.2f} "
                        f"Best Fold Val Loss {best_loss_across_folds:0.2f}"
                    )

                    loop_counter += 1

        # Free memory after model selection
        del train_loader, val_loader
        del model
        del optimizer
        del scheduler
        ################################
        #       MODEL ASSESSMENT      #
        ###############################

        best_i = np.argmax(result_dict["best_val_acc"])
        best_config = result_dict["config"][best_i]
        best_val_acc = result_dict["best_val_acc"][best_i]
        log.print_and_log(
            f"Winner of Model Selection | "
            f"hidden dim: {best_config['dim_embedding']} | "
            f"num_layers {best_config['num_layers']} | "
            f"lr {best_config['lr']}"
        )
        log.print_and_log(f"Winner Best Val Accuracy {result_dict['best_val_acc'][best_i]:0.2f}")

        train_loader, val_loader, test_loader = get_train_val_test_loaders(
            root,
            args.dataset_name,
            train_index,
            val_index,
            test_index,
            best_config["num_layers"],
            args.k,
            args.use_multi_gpu,
        )

        model_config = {
            "dim_embedding": best_config["dim_embedding"],
            "num_layers": best_config["num_layers"],
            "dim_features": features_dim,
            "dim_target": n_classes,
            "dropout": 0.5,
            "train_eps": True,
            "aggregation": "mean",
            "combine_multi_height": args.combine_multi_height,
            "use_linear_after_readout": args.use_linear_after_readout,
        }

        loop_counter = 1
        test_accuracies = []
        for _ in range(args.runs):
            model = DAGMLP(**model_config)
            reset_to_kaiming_uniform(model)

            if args.use_multi_gpu:
                number_of_available_gpus = cuda.device_count()
                model = DataParallel(model, device_ids=[*range(number_of_available_gpus)])

            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=best_config["lr"])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

            save_path = os.path.join("models", args.dataset_name, "model_best_" + logger_name + ".pth.tar")
            early_stopper = Patience(patience=args.patience, use_loss=False, save_path=save_path)

            pbar_train = tqdm(range(args.epochs), desc="Epoch 0 Loss 0")

            for epoch in pbar_train:
                train_acc, train_loss, val_acc, val_loss = train_and_validate_model(
                    model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.use_multi_gpu
                )
                if args.tensorboard:
                    writer.add_scalar(
                        f"Fold {it + 1} - "
                        f"(B) Model Assessment/run-{loop_counter}/Train Loss", train_loss, epoch
                    )
                    writer.add_scalar(
                        f"Fold {it + 1} - "
                        f"(B) Model Assessment/run-{loop_counter}/Train Accuracy", train_acc, epoch
                    )
                    writer.add_scalar(
                        f"Fold {it + 1} - "
                        f"(B) Model Assessment/run-{loop_counter}/Val Loss", val_loss, epoch
                    )
                    writer.add_scalar(
                        f"Fold {it + 1} - "
                        f"(B) Model Assessment/run-{loop_counter}/Val Accuracy", val_acc, epoch
                    )
                if early_stopper.stop(epoch, val_loss, val_acc, model=model):
                    break

                pbar_train.set_description(
                    f"Test {loop_counter}/{args.runs} "
                    f"Epoch {epoch + 1} "
                    f"Val acc {val_acc:0.1f} "
                    f"Best Epoch {early_stopper.best_epoch} "
                    f"Best Val Acc {early_stopper.val_acc:0.2f}"
                )

            checkpoint = load(save_path)
            epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])

            test_loss = 0
            test_count = 0
            test_correct = 0
            model.eval()

            for idx, data in enumerate(test_loader):
                if not args.use_multi_gpu:
                    data = data.to(device)
                y = cat([d.y for d in data], dim=0) if args.use_multi_gpu else data.y
                output, loss = eval(model, data, y, criterion)
                test_loss += loss.item() * output.size(0)
                test_count += output.size(0)
                prediction = output.max(1)[1].type_as(y)
                test_correct += prediction.eq(y.double()).sum().item()

            test_accuracy = (test_correct / test_count) * 100
            log.print_and_log(
                f"Test {loop_counter}/{args.runs} "
                f"Epoch {epoch + 1} "
                f"Val loss {val_loss:0.2f} "
                f"Val acc {val_acc:0.1f} "
                f"Best Val Loss {early_stopper.val_loss:0.2f} "
                f"Best Val Acc {early_stopper.val_acc:0.2f} "
                f"Test acc {test_accuracy:0.2f}"
            )
            if args.tensorboard:
                writer.add_scalar(f"Fold {it + 1} - "
                                  f"(C) Final Model Test/Accuracy run-{loop_counter}", test_accuracy, 0)

            loop_counter += 1
            test_accuracies.append(test_accuracy)

        all_accuracies_folds.append(test_accuracies)
        mean_accuracies_folds.append(np.mean(test_accuracies))
        best_config_across_folds.append(best_config)
        log.print_and_log(f"Test acc mean {mean_accuracies_folds[-1]:.3f}")

        log.print_and_log(
            f"Cross-val iter:{it + 1} | Current average test accuracy across folds {np.mean(mean_accuracies_folds):.5f}"
        )
        log.print_and_log("\n")

    result_dict = {
        "all_test_accuracies": all_accuracies_folds,
        "mean_test_accuracies": mean_accuracies_folds,
        "best_params": best_config_across_folds,
    }

    results_path = os.path.join("results", args.dataset_name, "results_" + logger_name + ".json")
    with open(results_path, "w") as f:
        json.dump(result_dict, f)

    log.print_and_log(f"AVERAGE TEST ACC ACROSS FOLDS {np.mean(mean_accuracies_folds):.5f}")
    log.print_and_log(f"STD ACROSS FOLDS {np.std(mean_accuracies_folds)}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="CanonizedDAGMLP")
    parser.add_argument("--dataset-name", default="IMDB-BINARY", type=str)
    parser.add_argument("--number-of-layers", default=[5], nargs="+", type=int)
    parser.add_argument("--build-only", action="store_true", default=False)
    parser.add_argument("--dim-embeddings",default=[64], nargs="+", type=int)
    parser.add_argument("--lrs", default=[1e-3], nargs="+", type=float)
    parser.add_argument("--combine-multi-height", action="store_true", default=False)
    parser.add_argument("--use-linear-after-readout", action="store_true", default=False)
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--batch-size", type=int, default=128, metavar="N")
    parser.add_argument("--runs", type=int, default=2, metavar="N")
    parser.add_argument("--epochs", type=int, default=500, metavar="N")
    parser.add_argument("--step-size", type=int, default=100, metavar="N")
    parser.add_argument("--patience", default=250, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--use-multi-gpu", action="store_true", default=False)
    parser.add_argument("--tensorboard", action="store_true", default=True)
    args = parser.parse_args()

    main(args)
