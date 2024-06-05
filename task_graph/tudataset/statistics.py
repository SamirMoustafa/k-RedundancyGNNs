import argparse
import os
from pathlib import Path

from tabulate import tabulate
from torch import load
from torch_geometric.datasets import TUDataset


def main(args):
    table_headers = ["Dataset", "Number of layers", "Number of nodes", "Number of edges"]
    table_data = []

    root = Path(__file__).resolve().parent.parent.parent.joinpath("data")
    number_of_layers = args.number_of_layers
    original_dataset = TUDataset(root=root, name=args.dataset_name)
    total_number_of_nodes = sum([data.num_nodes for data in original_dataset])
    total_number_of_edges = sum([data.num_edges for data in original_dataset])

    for num_layers in number_of_layers:
        table_data += [
            [
                args.dataset_name,
                num_layers,
                total_number_of_nodes * (num_layers + 1),
                total_number_of_edges * (num_layers + 1),
            ]
        ]

    for num_layers in number_of_layers:
        for k in args.k:
            canonized_dataset_path = os.path.join(
                root, args.dataset_name, f"canonized_num_layers_{num_layers}_k_{k}.pkl"
            )
            if not os.path.exists(canonized_dataset_path):
                raise ValueError(
                    f"Canonized dataset for {args.dataset_name} with {num_layers} layers and k={k} "
                    f"does not exist, or does not found in {canonized_dataset_path}. "
                    f"Use the script tu_dataset_run.py to build the canonized dataset."
                )
            else:
                print(f"Canonized dataset for {args.dataset_name} with {num_layers} layers and k={k} Found!")

            dataset_list = load(canonized_dataset_path)
            total_number_of_nodes = sum([data.x.shape[0] for data in dataset_list])
            total_number_of_edges = sum([data.dag_edge_index.shape[1] for data in dataset_list])
            table_data += [[f"{args.dataset_name} {k}-NTs", num_layers, total_number_of_nodes, total_number_of_edges]]

    table = tabulate(table_data, headers=table_headers, tablefmt="github")
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="MUTAG")
    parser.add_argument("--number_of_layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--k", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    main(args)
