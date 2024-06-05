import argparse
import datetime
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from joblib import Parallel, delayed
from torch import finfo, nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.dag_gnn import ToCanonizedDirectedAcyclicGraph
from task_graph.synthetic.dagmlp_model import DAGMLP
from task_graph.synthetic.dataset.CSL import CSL
from task_graph.synthetic.dataset.EXP import EXP
from task_graph.synthetic.loader import Dataset, prepare_loaders
from utils import PrinterLogger

DATASETS = {
    "CSL": CSL,
    "EXP": EXP,
    "CEXP": EXP,
}

DATASET_NAMES = {
    "CSL": ["CSL"],
    "EXP": ["EXP"],
    "CEXP": ["CEXP"],
}


def train_step(model, data, criterion, optimizer):
    output = model(data)
    loss_train = criterion(output, data.y)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    return output, loss_train


@torch.no_grad()
def eval_step(model, data, criterion):
    output = model(data)
    loss_test = criterion(output, data.y)
    return output, loss_test


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_count = 0
    for data in loader:
        data = data.to(device)
        output, loss = train_step(model, data, criterion, optimizer)
        total_loss += loss.item() * data.num_graphs
        preds = output.argmax(dim=1).type_as(data.y)
        correct_count += preds.eq(data.y).sum().item()
    return total_loss / len(loader.dataset), correct_count / len(loader.dataset)


def evaluate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct_count = 0
    for data in loader:
        data = data.to(device)
        output, loss = eval_step(model, data, criterion)
        total_loss += loss.item() * data.num_graphs
        preds = output.argmax(dim=1).type_as(data.y)
        correct_count += preds.eq(data.y).sum().item()
    return total_loss / len(loader.dataset), correct_count / len(loader.dataset)


def run_classification(model, loaders, epochs=200, lr=1e-2):
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    pbar_train = tqdm(range(epochs), desc="Training Progress")
    for epoch in pbar_train:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion)

        pbar_train.set_description(
            f"Epoch {epoch + 1} Train Loss {train_loss:.3f} Train Acc {train_acc:.3f} Val Loss {val_loss:.3f} Val Acc {val_acc:.3f}"
        )

    train_acc = evaluate_one_epoch(model, train_loader, criterion)[1]
    val_acc = evaluate_one_epoch(model, val_loader, criterion)[1]
    test_acc = evaluate_one_epoch(model, test_loader, criterion)[1]

    log.print_and_log(f"Train Acc {train_acc:.2f} Val Acc {val_acc:.2f} Test Acc {test_acc:.2f}")

    return train_acc, val_acc, test_acc


def main_classification(dataset, model_config, train_config):
    splits = dataset.get_all_splits_idx()
    results = defaultdict(list)
    n_splits = dataset.n_splits
    model = DAGMLP(**model_config).to(device)

    for index in range(n_splits):
        model.reset_parameters()
        log.print_and_log(f"Split {index + 1}/{n_splits}")

        loaders = prepare_loaders(dataset, splits, batch_size=train_config["batch_size"], index=index)
        train_acc, val_acc, test_acc = run_classification(
            model, loaders, epochs=train_config["epochs"], lr=train_config["lr"]
        )

        results["train"].append(train_acc)
        results["val"].append(val_acc)
        results["test"].append(test_acc)

    return {k: np.asarray(v) for k, v in results.items()}


def _isomorphism(predictions, eps=None, p=2):
    # Return the failure percentage... the smaller, the better!
    eps = eps or 2.0 * float(finfo(torch.float64).eps)
    predictions = torch.tensor(predictions, dtype=torch.float64)
    mm = torch.pdist(predictions, p=p)
    wrong = (mm < eps).sum().item()
    metric = wrong / mm.shape[0]
    return metric


def main_isomorphism(dataset, model_config, train_config):
    loader = DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=False)
    model = DAGMLP(predict=False, **model_config).to(device)
    res = []
    for i in range(5):
        torch.manual_seed(i)
        model.reset_parameters()
        embeddings, lst = [], []
        model.eval()
        for data in tqdm(loader):
            pre = model(data.to(device))
            embeddings.append(pre.detach().cpu())
        print(f"Failure Rate : {_isomorphism(torch.cat(embeddings, 0).detach().cpu().numpy())}")
        res.append(_isomorphism(torch.cat(embeddings, 0).detach().cpu().numpy()))

    return {"n_sim_pairs": np.asarray(res)}


def setup_directories(dataset_name):
    for folder_type in ["results", "logs", "models"]:
        path = os.path.join(os.getcwd(), folder_type, dataset_name)
        if not os.path.exists(path):
            os.makedirs(path)


def get_config(args):
    # Dataset Name | Maximum Diameter
    # -------------------------------
    # CSL          | 10
    # CEXP         | 28
    # EXP          | 17

    config_dict = {
        "CSL": {
            "task": "classification",
            "model_config": {
                "dim_features": 1,
                "dim_embedding": 64,
                "dim_target": 10,
                "num_layers": 6,
            },
            "train_config": {
                "batch_size": 32,
                "epochs": 200,
                "lr": 1e-3,
            },
        },
        "CEXP": {
            "task": "classification",
            "model_config": {
                "dim_features": 1,
                "dim_embedding": 64,
                "dim_target": 2,
                "num_layers": 15,
            },
            "train_config": {
                "batch_size": 32,
                "epochs": 200,
                "lr": 1e-3,
            },
        },
        "EXP": {
            "task": "iso",
            "model_config": {
                "dim_features": 1,
                "dim_embedding": 1,
                "dim_target": 1,
                "num_layers": 6,
            },
            "train_config": {
                "batch_size": 1,
            },
        },
    }
    return config_dict.get(args.dataset, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN baselines on synthetic datasets")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="CSL")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--skip_experiment", action="store_true", default=False)
    args = parser.parse_args()

    config = get_config(args)
    if config is None:
        raise ValueError("Dataset not supported")

    setup_directories(args.dataset)

    device = torch.device(args.device)
    now = "_" + "-".join(str(datetime.datetime.today()).split()).split(".")[0].replace(":", ".")

    logging.basicConfig(
        filename=os.path.join("logs", args.dataset, args.dataset + now + ".log"), level=logging.INFO, filemode="w"
    )
    log = PrinterLogger(logging.getLogger(__name__))
    results = defaultdict(dict)
    task = config["task"]

    log.print_and_log("PROCESSING ...")
    for d_name in DATASET_NAMES[args.dataset]:
        d_name = str(d_name)
        log.print_and_log(f"Model Name {d_name.upper()}")
        dataset = DATASETS[args.dataset](task=task) if "EXP" == d_name else DATASETS[args.dataset]()

        # Comment the follow to remove the canonization step
        num_layers = config["model_config"]["num_layers"]
        dataset_directory = dataset.root
        canonized_file_name = d_name.split(".")[0] + f"_num_layers_{num_layers}_k_{args.k}.pt"
        canonized_file_path = os.path.join(dataset_directory, canonized_file_name)
        if os.path.exists(canonized_file_path):
            log.print_and_log(f"Loading canonized dataset from {canonized_file_path}")
            data_list = torch.load(canonized_file_path)
        else:
            dag_transform = ToCanonizedDirectedAcyclicGraph(
                num_nodes=None, num_layers=num_layers, k=args.k, n_jobs=1, verbose=False
            )
            if args.n_jobs == 1:
                data_list = [dag_transform(g) for g in tqdm(dataset)]
            else:
                data_list = Parallel(n_jobs=-1)(delayed(dag_transform)(g) for g in tqdm(dataset))
            torch.save(data_list, canonized_file_path)

        if not args.skip_experiment:
            if task == "classification":
                split_func = dataset.get_all_splits_idx
                n_splits = dataset.n_splits

            dataset = Dataset(data_list)

            if task == "classification":
                dataset.get_all_splits_idx = split_func
                dataset.n_splits = n_splits

            # Comment the above to remove the canonization step
            if task == "classification":
                results[d_name] = main_classification(dataset, config["model_config"], config["train_config"])
                log.print_and_log(
                    f"Train Avg {results[d_name]['train'].mean():0.4f} Train Std {results[d_name]['train'].std():0.4f}"
                )
                log.print_and_log(
                    f"Val Avg   {results[d_name]['val'].mean():0.4f} Val Std   {results[d_name]['val'].std():0.4f}"
                )
                log.print_and_log(
                    f"Test Avg  {results[d_name]['test'].mean():0.4f} Test Std  {results[d_name]['test'].std():0.4f}"
                )
            elif task == "iso":
                results[d_name] = main_isomorphism(dataset, config["model_config"], config["train_config"])
                log.print_and_log(f"Avg # of similar pairs: {results[d_name]['n_sim_pairs'].mean()}")
                log.print_and_log(f"Std # of similar pairs: {results[d_name]['n_sim_pairs'].std()}")
            else:
                raise ValueError("Task not supported")

            with open(os.path.join("results", args.dataset + "_results.pkl"), "wb") as f:
                pickle.dump(results, f)
