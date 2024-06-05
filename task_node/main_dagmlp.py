from utils import seed_everything

seed_everything(42)

import argparse
import itertools
import logging
from datetime import datetime
from os import makedirs, mkdir
from os.path import dirname, exists, join, realpath
from pathlib import Path
from time import time

from numpy import mean, std
from prettytable import PrettyTable
from torch import cuda
from torch import device as torch_device
from torch import load, no_grad, save
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm

from src.dag_gnn import ToCanonizedDirectedAcyclicGraph
from task_node.dagmlp_model import DAGMLP


class SliceMaskData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._train_mask, self._val_mask, self._test_mask = (
            kwargs["train_mask"],
            kwargs["val_mask"],
            kwargs["test_mask"],
        )
        assert len(self._train_mask.shape) == 2, "All masks must be 2D"
        assert (
            self._train_mask.shape == self._val_mask.shape == self._test_mask.shape
        ), "All masks must have the same shape"
        self.number_of_splits = self._train_mask.shape[1]
        self.split = 0

    def increase_split(self):
        self.split = (self.split + 1) % self.number_of_splits

    @property
    def train_mask(self):
        return self._train_mask[:, self.split]

    @property
    def val_mask(self):
        return self._val_mask[:, self.split]

    @property
    def test_mask(self):
        return self._test_mask[:, self.split]


def setup_logger(filename, verbose, logger_name=None):
    # If no logger_name is provided, use the module name.
    name = logger_name or __name__
    # Create a logger object
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # If the logger already has handlers, clear them
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    if verbose:
        # Create a stream handler to print logs to console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    if verbose:
        stream_handler.setFormatter(formatter)
    # Add handlers to logger
    logger.addHandler(file_handler)
    if verbose:
        logger.addHandler(stream_handler)
    return logger


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    log_probs = model(data)
    loss = cross_entropy(log_probs[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@no_grad()
def validate(model, data):
    model.eval()
    log_probs = model(data)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    datasets_names = ["Cornell", "Texas", "Wisconsin", "Cora", "CiteSeer", "PubMed"]
    table_header = ["Model"] + datasets_names
    table = PrettyTable(table_header)
    experiment_id = 0
    choices_layers = [2, 3]
    choices_k = [0, 1]
    choices_combine_functions = [None, "sum", "mean", "cat"]
    total_number_of_choices = 3 + 1
    for default_num_layers, default_k, default_combine_function in itertools.product(
        *(choices_layers, choices_k, choices_combine_functions)
    ):
        row_i = [
            f"DAGMLP (l={default_num_layers}; k={default_k}) - {str(default_combine_function).upper()} Combine",
        ]
        for default_dataset_name in datasets_names:
            parser = argparse.ArgumentParser()
            parser.add_argument("--dataset_name", type=str, default=default_dataset_name)
            parser.add_argument("--num_layers", type=int, default=default_num_layers)
            parser.add_argument("--k", type=lambda x: int(x) if x.isnumeric() else None, default=f"{default_k}")
            parser.add_argument("--build_only", action="store_true")
            parser.add_argument("--num_of_runs", type=int, default=10)
            parser.add_argument("--epochs", type=int, default=1000)
            parser.add_argument("--dim_embeddings", type=int, default=128)
            parser.add_argument("--train_eps", action="store_true", default=True)
            parser.add_argument("--lr", type=float, default=1e-2)
            parser.add_argument("--weight_decay", type=float, default=5e-6)
            parser.add_argument("--device_id", type=int, default=0)
            parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
            parser.add_argument("--verbose", type=str2bool, default=True)
            parser.add_argument("--logging_dir", type=str, default="./logs")
            args = parser.parse_args()

            device = torch_device(args.device if args.device == "cpu" else f"cuda:{args.device_id}")

            path = join(Path(dirname(realpath(__file__))).parent.parent, "data", f"undirected_{args.dataset_name}")
            if args.dataset_name in ["Cora", "CiteSeer", "PubMed"]:
                dataset = Planetoid(path, args.dataset_name)
            elif args.dataset_name in ["Cornell", "Texas", "Wisconsin"]:
                dataset = WebKB(path, args.dataset_name, transform=ToUndirected())
            else:
                raise ValueError(f"Dataset {args.dataset_name} not supported")
            data = dataset[0]

            canonized_dataset_path = join(path, f"canonized_num_layers_{args.num_layers}_k_{args.k}.pkl")
            if not exists(canonized_dataset_path):
                makedirs(path, exist_ok=True)
                print(
                    f"Building canonized dataset for {args.dataset_name} "
                    f"with {args.num_layers} layers and {args.k} k"
                )
                dag_transform = ToCanonizedDirectedAcyclicGraph(None, args.num_layers, args.k, 1, True)
                start_time = time()
                data = dag_transform(data)
                end_time = time()
                runtime_path = join(path, f"runtime_num_layers_{args.num_layers}_k_{args.k}.txt")
                with open(runtime_path, "w") as f:
                    f.write(str(end_time - start_time))
                save(data, canonized_dataset_path)
                print(
                    f"Canonized dataset for {args.dataset_name} "
                    f"with {args.num_layers} layers and k={args.k} saved in {canonized_dataset_path}"
                )
            else:
                print(
                    f"Canonized dataset for {args.dataset_name} "
                    f"with {args.num_layers} layers and k={args.k} already exists"
                )
                data = load(canonized_dataset_path)

            if not args.build_only:
                num_features = dataset.num_features
                num_classes = dataset.num_classes

                model_kwargs = {
                    "dim_features": num_features,
                    "dim_embedding": args.dim_embeddings,
                    "dim_target": num_classes,
                    "dropout": 0.5,
                    "train_eps": args.train_eps,
                    "num_layers": args.num_layers,
                    "combine_multi_height": True if default_combine_function is not None else False,
                    "combine_function": default_combine_function,
                }

                if args.dataset_name in ["Cornell", "Texas", "Wisconsin"]:
                    data = SliceMaskData(**data)
                data = data.to(device)

                model = DAGMLP(**model_kwargs).to(device)
                model_name = model.__class__.__name__
                optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                logging_dir = join(args.logging_dir, args.dataset_name, model_name)
                if not exists(logging_dir):
                    if not exists(join(args.logging_dir, args.dataset_name)):
                        if not exists(join(args.logging_dir)):
                            mkdir(args.logging_dir)
                        mkdir(join(args.logging_dir, args.dataset_name))
                    mkdir(logging_dir)

                current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
                logger = setup_logger(
                    filename=f"{logging_dir}/num_layers{args.num_layers}-k{args.k}-{current_time}.log",
                    verbose=args.verbose,
                )

                logger.info(vars(args))
                logger.info(model_kwargs)

                accuracies = []
                for i in range(args.num_of_runs):
                    model = DAGMLP(**model_kwargs).to(device)
                    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                    pbar = tqdm(range(args.epochs))
                    best_epoch, best_val_acc, best_test_acc, best_model_state_dict = 0, 0, 0, None
                    model.reset_parameters()
                    # Evaluate model performance (accuracy)
                    for epoch in pbar:
                        loss = train(model, optimizer, data)
                        train_acc, val_acc, test_acc = validate(model, data)
                        if val_acc > best_val_acc:
                            best_epoch, best_val_acc, best_test_acc, best_model_state_dict = (
                                epoch,
                                val_acc,
                                test_acc,
                                model.state_dict(),
                            )
                        else:
                            best_epoch, best_val_acc, best_test_acc, best_model_state_dict = (
                                best_epoch,
                                best_val_acc,
                                best_test_acc,
                                best_model_state_dict,
                            )
                        pbar.set_description(
                            f"Mean test: {mean(accuracies):.0f}% | "
                            f"{i + 1}/{args.num_of_runs} | "
                            f"Best (Epoch: {best_epoch}, "
                            f"Test: {best_test_acc * 100:.1f}%) | "
                            f"Loss: {loss:.3f}"
                        )
                    logger.info(
                        f"Mean test: {mean(accuracies):.0f}% | "
                        f"{i + 1}/{args.num_of_runs} | "
                        f"Best (Epoch: {best_epoch}, "
                        f"Test: {best_test_acc * 100:.1f}%)"
                    )
                    accuracies.append(best_test_acc * 100)
                    if isinstance(data, SliceMaskData):
                        data.increase_split()

                results = f"{mean(accuracies):.2f} ± {std(accuracies):.1f}"
                logger.info(
                    f"{args.dataset_name} - "
                    f"{model_name} num_layers: {args.num_layers} k: {args.k} | "
                    f"accuracy: {results}"
                )

            row_i.append(results)
        experiment_id += 1
        table.add_row(row_i)
        if not experiment_id % total_number_of_choices:
            table.add_row(["-" * 40] * (len(datasets_names) + 1))
        print(table)
