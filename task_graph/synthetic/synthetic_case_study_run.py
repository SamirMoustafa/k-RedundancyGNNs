import os
from collections import defaultdict

import numpy as np
import torch
from joblib import Parallel, delayed
from prettytable import PrettyTable
from torch import nn
from tqdm import tqdm

from src.dag_gnn import ToCanonizedDirectedAcyclicGraph
from task_graph.synthetic.dagmlp_model import DAGMLP
from task_graph.synthetic.dataset.CSL import CSL
from task_graph.synthetic.loader import Dataset, prepare_loaders


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

    best_val_acc, best_model_state_dict = 0, None
    pbar_train = tqdm(range(epochs), desc="Training Progress")
    for epoch in pbar_train:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state_dict = model.state_dict()
        pbar_train.set_description(
            f"Epoch {epoch + 1} "
            f"Train Loss {train_loss:.3f} "
            f"Train Acc {train_acc:.3f} "
            f"Val Loss {val_loss:.3f} "
            f"Val Acc {val_acc:.3f}"
        )

    model.load_state_dict(best_model_state_dict)
    train_acc = evaluate_one_epoch(model, train_loader, criterion)[1]
    val_acc = evaluate_one_epoch(model, val_loader, criterion)[1]
    test_acc = evaluate_one_epoch(model, test_loader, criterion)[1]
    return train_acc, val_acc, test_acc


def main_classification(dataset, model_config, train_config):
    splits = dataset.get_all_splits_idx()
    results = defaultdict(list)
    n_splits = dataset.n_splits
    model = DAGMLP(**model_config).to(device)

    for index in range(n_splits):
        model.reset_parameters()

        loaders = prepare_loaders(dataset, splits, batch_size=train_config["batch_size"], index=index)
        train_acc, val_acc, test_acc = run_classification(
            model, loaders, epochs=train_config["epochs"], lr=train_config["lr"]
        )

        results["train"].append(train_acc)
        results["val"].append(val_acc)
        results["test"].append(test_acc)

    return {k: np.asarray(v) for k, v in results.items()}


def run_inner_loop(k, num_layers, model_config, train_config):
    if k > num_layers:
        return "-"
    else:
        model_config.update({"num_layers": num_layers})
        dataset = CSL()
        # Comment the follow to remove the canonization step
        dataset_directory = dataset.root
        canonized_file_name = f"_num_layers_{num_layers}_k_{k}.pt"
        canonized_file_path = os.path.join(dataset_directory, canonized_file_name)
        if os.path.exists(canonized_file_path):
            data_list = torch.load(canonized_file_path)
        else:
            dag_transform = ToCanonizedDirectedAcyclicGraph(
                num_nodes=None, num_layers=num_layers, k=k, n_jobs=1, verbose=False
            )
            data_list = Parallel(n_jobs=-1)(delayed(dag_transform)(g) for g in tqdm(dataset))
            torch.save(data_list, canonized_file_path)

        split_func = dataset.get_all_splits_idx
        n_splits = dataset.n_splits

        dataset = Dataset(data_list)
        dataset.get_all_splits_idx = split_func
        dataset.n_splits = n_splits

        test_accuracy = main_classification(dataset, model_config, train_config)["test"] * 100
        test_accuracy = f"{test_accuracy.mean().round(1)} ± {test_accuracy.std().round(1)}"
        return test_accuracy


if __name__ == "__main__":
    max_num_layers = 6
    model_config = {
        "dim_features": 1,
        "dim_embedding": 64,
        "dim_target": 10,
    }
    train_config = {
        "batch_size": 32,
        "epochs": 200,
        "lr": 1e-3,
    }

    device = torch.device("cuda")
    task = "classification"

    table_header = ["K's"] + [f"{l} layers" for l in range(1, max_num_layers + 1)]
    table = PrettyTable(table_header)

    for k in range(max_num_layers + 1):
        test_accuracies_per_k = [
            run_inner_loop(k, num_layers, model_config, train_config) for num_layers in range(1, max_num_layers + 1)
        ]
        table.add_row(
            [
                f"{k}-NTs",
            ]
            + test_accuracies_per_k
        )
        print(table)
