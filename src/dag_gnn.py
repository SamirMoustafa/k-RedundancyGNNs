from collections import OrderedDict
from itertools import chain
from typing import Dict, List

from numpy import ndarray
from torch import Tensor, arange, float32, isin, stack, tensor
from torch_geometric.data import Batch, Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from src.canonized_dag import CanonizedDAG
from src.dag import DAG, find_smallest_missing_integer
from src.hash_function import inverse_hash
from src.tensor_dag import BatchTensorCanonizedDAG, TensorCanonizedDAG, TensorDAG, add_graphs


class NoneCanonizedDirectedAcyclicGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs) -> int:
        if "dag_layers_mask" in key:
            return 0
        if "dag_edge_index" in key or "dag_leaves" in key or "dag_roots" in key:
            return self.dag_num_nodes
        if "edge_index" in key or "dag_nodes_order" in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs) -> int:
        if "index" in key or "indices" in key:
            return 1
        if "dag_roots" in key or "dag_leaves" in key:
            return -1
        else:
            return 0


class CanonizedDirectedAcyclicGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs) -> int:
        if "dag_layers_mask" in key:
            return 0
        if "dag_edge_index" in key or "readout" in key or "leaves" in key:
            return self.num_nodes
        if "edge_index" in key:
            return self.original_graph_num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs) -> int:
        if "index" in key or "indices" in key:
            return 1
        if "readout" in key or "leaves" in key:
            return -1
        else:
            return 0


def get_isolated_nodes_indices(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Returns the indices of the isolated nodes those don't exist in the edge_index tensor.

    :param edge_index: Edge index tensor
    :param num_nodes: Number of nodes in the entire graph
    :return: Indices of the isolated nodes that don't exist in the edge_index tensor
    """
    device = edge_index.device
    isolated_nodes = arange(num_nodes).to(device)
    isolated_nodes = isolated_nodes[~isin(isolated_nodes, edge_index.unique())]
    return isolated_nodes.tolist()


def labels_for_isolated_nodes(labels: dict, edge_index: Tensor, num_nodes: int) -> dict:
    """
    Adds labels for isolated nodes to a dictionary of labels.

    :param labels: Dictionary of labels
    :param edge_index: Edge index tensor
    :param num_nodes: Number of nodes
    :return: Dictionary of labels with labels for isolated nodes added
    """
    if len(labels) == 0:
        return labels
    isolated_nodes = get_isolated_nodes_indices(edge_index, num_nodes)
    isolated_nodes_labels = dict()
    for isolated_node in isolated_nodes:
        free_id = find_smallest_missing_integer(labels.keys())
        labels[free_id] = str(isolated_node)
        isolated_nodes_labels[free_id] = str(isolated_node)
    return isolated_nodes_labels


@functional_transform("to_directed_acyclic_graph")
class ToNoneCanonizedDirectedAcyclicGraph(BaseTransform):
    """
    Transforms a graph into a directed acyclic graph
    """

    def __init__(
        self, num_nodes: int | None, num_layers: int, k: int = 0, n_jobs: int = -1, verbose: bool = True, **kwargs
    ):
        self.num_nodes: int = num_nodes
        self.num_layers: int = num_layers
        self.k: int = k
        self.n_jobs: int = n_jobs
        self.verbose: bool = verbose

    def __call__(self, data: Data) -> NoneCanonizedDirectedAcyclicGraphData:
        for i, store in enumerate(data.edge_stores):
            # Skip if there is no edge_index
            if "edge_index" not in store:
                continue
            num_nodes = self.num_nodes if self.num_nodes is not None else store.num_nodes
            num_nodes = num_nodes if num_nodes is not None else store.edge_index.max().item() + 1

            tensor_dag: TensorDAG = TensorDAG(num_nodes, self.num_layers, self.k, self.n_jobs, self.verbose)
            dag_edge_index: Tensor = tensor_dag(store.edge_index)
            dag: DAG = tensor_dag.get_dag()
            dag_labels: Dict = dag.labels.copy()
            isolated_nodes_labels: Dict = labels_for_isolated_nodes(dag_labels, store.edge_index, num_nodes)
            entire_dag_labels: Dict = OrderedDict(sorted({**dag_labels, **isolated_nodes_labels}.items()))

            dag_layers_mask: Tensor = tensor_dag.compute_masked_indices(dag_edge_index)
            dag_nodes_order: Tensor = tensor(list(map(int, entire_dag_labels.values())))
            dag_roots: Tensor = tensor(
                sorted(
                    list(dag.get_roots()) + list(isolated_nodes_labels.keys()), key=lambda x: int(entire_dag_labels[x])
                )
            )
            dag_leaves: Tensor = tensor(
                sorted(
                    list(dag.get_leaves()) + list(isolated_nodes_labels.keys()), key=lambda x: int(entire_dag_labels[x])
                )
            )
            dag_num_nodes: int = len(dag_edge_index.unique()) + len(isolated_nodes_labels)

            keys = [
                "num_nodes",
                "dag_edge_index",
                "dag_layers_mask",
                "dag_nodes_order",
                "dag_roots",
                "dag_leaves",
                "dag_num_nodes",
            ]
            values = [num_nodes, dag_edge_index, dag_layers_mask, dag_nodes_order, dag_roots, dag_leaves, dag_num_nodes]

            for key, value in store.items():
                if key in keys:
                    continue
                if store.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)
            store.update(zip(keys, values))
        data = NoneCanonizedDirectedAcyclicGraphData(**data.to_dict())
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(num_layers={self.num_layers}, k={self.k})"


def element_or_row_wise_multiply(matrix, vector):
    # Check if the tensor is a vector (1D) or a matrix (2D)
    if len(matrix.shape) == 1:
        # Element-wise multiplication for vectors
        x = matrix * vector
    elif len(matrix.shape) == 2:
        # Row-wise multiplication for matrices
        vector = vector.unsqueeze(1)
        x = matrix * vector
    else:
        raise ValueError("Input tensor must be either 1D or 2D.")
    return x


@functional_transform("to_canonized_directed_acyclic_graph")
class ToCanonizedDirectedAcyclicGraph(ToNoneCanonizedDirectedAcyclicGraph):
    def __init__(self, num_nodes: int | None, num_layers: int, k: int = 0, n_jobs: int = -1, verbose: bool = True):
        super().__init__(num_nodes, num_layers + 1, k, n_jobs, verbose)

    def __call__(self, data: Data) -> CanonizedDirectedAcyclicGraphData:
        for i, store in enumerate(data.edge_stores):
            # Skip if there is no edge_index
            if "edge_index" not in store:
                continue

            # Extract node and edge attributes
            node_attributes = store.x if "x" in store else [None for _ in range(store.num_nodes)]
            edge_attributes = store.edge_attr if "edge_attr" in store else None

            # Extract number of nodes
            num_nodes = self.num_nodes if self.num_nodes is not None else store.num_nodes
            num_nodes = num_nodes if num_nodes is not None else store.edge_index.max().item() + 1
            # Create canonized tensor dag
            canonized_tensor_dag: TensorCanonizedDAG = TensorCanonizedDAG(
                num_nodes, self.num_layers, self.k, self.n_jobs, self.verbose
            )
            canonized_tensor_dag(store.edge_index, node_attributes, edge_attributes)
            canonized_dag: CanonizedDAG = canonized_tensor_dag.get_dag()

            # Extract canonized dag properties
            dag_x: List = [inverse_hash(h) for h in canonized_dag.h_node_attributes.values()]
            dag_x: Tensor = (
                None if all(item is None for item in dag_x) else stack([tensor(dag_x_i) for dag_x_i in dag_x])
            )

            edge_multiplicities: Dict[tuple, List[int]] = {
                e: [*e_m_dict.values()] for e, e_m_dict in canonized_dag.edge_multiplicities.items()
            }
            edge_multiplicities: Tensor = tensor(list(chain.from_iterable(edge_multiplicities.values())), dtype=float32)
            edge_attributes: Dict[tuple, ndarray] = {
                e: [inverse_hash(eh_i) for eh_i in e_h] for e, e_h in canonized_dag.h_edge_attributes.items()
            }
            has_edge_attributes = all(item is not None for item in edge_attributes.values())
            edge_attributes: Tensor = (
                None
                if not has_edge_attributes
                else stack(
                    [
                        tensor(edge_attr, dtype=float32)
                        for edge_attr in list(chain.from_iterable(edge_attributes.values()))
                    ]
                )
            )
            dag_edge_attr: Tensor = (
                edge_multiplicities
                if not has_edge_attributes
                else element_or_row_wise_multiply(edge_attributes, edge_multiplicities)
            )
            canonization_dag_edge_index = tensor(
                [e for e, e_h in canonized_dag.h_edge_attributes.items() for _ in e_h]
            ).t()
            canonization_dag_layers_mask = canonized_tensor_dag.compute_masked_indices(canonization_dag_edge_index)
            canonization_dag_readout_at_each_layer: Tensor = tensor(canonized_tensor_dag.readout_at_each_layer)
            canonization_dag_leaves_at_each_layer: Tensor = tensor(canonized_tensor_dag.leaves_at_each_layer)
            canonization_dag_num_nodes: int = dag_x.shape[0]

            for readout_at_layer_i in canonization_dag_readout_at_each_layer:
                assert (
                    readout_at_layer_i.shape[0] == num_nodes
                ), f"Canonization dag roots shape {readout_at_layer_i.shape} is not equal to num_nodes {num_nodes}"

            keys = [
                "num_nodes",
                "original_graph_num_nodes",
                "dag_edge_index",
                "dag_x",
                "dag_layers_mask",
                "dag_readout_at_each_layer",
                "dag_leaves_at_each_layer",
                "edge_multiplicities",
                "dag_edge_attr",
            ]
            values = [
                canonization_dag_num_nodes,
                num_nodes,
                canonization_dag_edge_index,
                dag_x,
                canonization_dag_layers_mask,
                canonization_dag_readout_at_each_layer,
                canonization_dag_leaves_at_each_layer,
                edge_multiplicities,
                dag_edge_attr,
            ]

        for key, value in data.items():
            if key in keys:
                continue
            keys.append(key)
            values.append(value)
        data = CanonizedDirectedAcyclicGraphData(**dict(zip(keys, values)))
        return data


@functional_transform("to_canonized_directed_acyclic_graph_batched")
class ToCanonizedDirectedAcyclicGraphBatched(ToCanonizedDirectedAcyclicGraph):
    def __init__(
        self, num_layers: int, k: int = 0, n_graphs_jobs: int = -1, n_dag_jobs: int = -1, verbose: bool = True
    ):
        super().__init__(None, num_layers, k, n_dag_jobs, verbose)
        self.n_jobs_graphs: int = n_graphs_jobs
        self.batched_tensor_dag: BatchTensorCanonizedDAG = BatchTensorCanonizedDAG(
            self.num_layers, self.k, n_graphs_jobs, n_dag_jobs, verbose
        )

        self.graph_id_to_canonized_dag: Dict[int, list] = dict()
        self.graph_id_to_node_to_root_canonization_at_each_layer: Dict[int, list] = dict()
        self.graph_id_to_node_to_leave_canonization_at_each_layer: Dict[int, list] = dict()

    def from_data_list(self, graph_ids: List[int], data_list: List[Data]) -> CanonizedDirectedAcyclicGraphData:
        # process graphs that are not in the cache
        graph_ids_to_process = [graph_id for graph_id in graph_ids if graph_id not in self.graph_id_to_canonized_dag]
        data_list_to_process = [
            data for graph_id, data in zip(graph_ids, data_list) if graph_id in graph_ids_to_process
        ]

        edge_indices: List[Tensor] = [data.edge_index for data in data_list_to_process]
        node_attributes: List[Tensor] = [data.x for data in data_list_to_process]
        edge_attributes: List[Tensor] = [data.edge_attr for data in data_list_to_process]

        if len(graph_ids_to_process) > 0:
            (
                dags_per_graph,
                node_to_root_canonization_at_each_layer_per_graph,
                node_to_leave_canonization_at_each_layer_per_graph,
            ) = add_graphs(
                edge_indices,
                node_attributes,
                edge_attributes,
                None,
                self.num_layers,
                self.k,
                self.n_jobs_graphs,
                self.n_jobs,
                self.verbose,
            )
            # cache canonized dags, node_to_root_canonization_at_each_layer and node_to_leave_canonization_at_each_layer
            self.graph_id_to_canonized_dag.update(dict(zip(*[graph_ids, dags_per_graph])))
            self.graph_id_to_node_to_root_canonization_at_each_layer.update(
                dict(zip(*[graph_ids, node_to_root_canonization_at_each_layer_per_graph]))
            )
            self.graph_id_to_node_to_leave_canonization_at_each_layer.update(
                dict(zip(*[graph_ids, node_to_leave_canonization_at_each_layer_per_graph]))
            )

        # retrieve canonized dags, node_to_root_canonization_at_each_layer and node_to_leave_canonization_at_each_layer
        dags_per_graph = [self.graph_id_to_canonized_dag[graph_id] for graph_id in graph_ids]
        node_to_root_canonization_at_each_layer_per_graph = [
            self.graph_id_to_node_to_root_canonization_at_each_layer[graph_id] for graph_id in graph_ids
        ]
        node_to_leave_canonization_at_each_layer_per_graph = [
            self.graph_id_to_node_to_leave_canonization_at_each_layer[graph_id] for graph_id in graph_ids
        ]

        self.batched_tensor_dag(
            dags_per_graph,
            node_to_root_canonization_at_each_layer_per_graph,
            node_to_leave_canonization_at_each_layer_per_graph,
        )

        exclude_keys = list(set(data_list[0].to_dict().keys()) - {"x", "edge_index", "y"})
        batch_data: Data = Batch.from_data_list(data_list, exclude_keys=exclude_keys)

        for i, store in enumerate(batch_data.edge_stores):
            canonized_dag: CanonizedDAG = self.batched_tensor_dag.get_dag()
            # Extract canonized dag properties
            dag_x: List = [inverse_hash(h) for h in canonized_dag.h_node_attributes.values()]
            dag_x: Tensor = (
                None if all(item is None for item in dag_x) else stack([tensor(dag_x_i) for dag_x_i in dag_x])
            )

            edge_multiplicities: Dict[tuple, List[int]] = {
                e: [*e_m_dict.values()] for e, e_m_dict in canonized_dag.edge_multiplicities.items()
            }
            edge_multiplicities: Tensor = tensor(list(chain.from_iterable(edge_multiplicities.values())), dtype=float32)
            edge_attributes: Dict[tuple, ndarray] = {
                e: [inverse_hash(eh_i) for eh_i in e_h] for e, e_h in canonized_dag.h_edge_attributes.items()
            }
            has_edge_attributes = all(item is not None for item in edge_attributes.values())
            edge_attributes: Tensor = (
                None
                if not has_edge_attributes
                else stack(
                    [
                        tensor(edge_attr, dtype=float32)
                        for edge_attr in list(chain.from_iterable(edge_attributes.values()))
                    ]
                )
            )
            dag_edge_attr: Tensor = (
                edge_multiplicities
                if not has_edge_attributes
                else element_or_row_wise_multiply(edge_attributes, edge_multiplicities)
            )
            canonization_dag_edge_index = tensor(
                [e for e, e_h in canonized_dag.h_edge_attributes.items() for _ in e_h]
            ).t()
            canonization_dag_layers_mask: Tensor = self.batched_tensor_dag.compute_masked_indices(
                canonization_dag_edge_index
            )
            canonization_dag_readout_at_each_layer: List[Tensor] = [
                tensor(graph_i_readout_at_each_layer)
                for graph_i_readout_at_each_layer in self.batched_tensor_dag.readout_at_each_layer
            ]
            canonization_dag_leaves_at_each_layer: List[Tensor] = [
                tensor(graph_i_leaves_at_each_layer)
                for graph_i_leaves_at_each_layer in self.batched_tensor_dag.leaves_at_each_layer
            ]
            canonization_dag_num_nodes: int = dag_x.shape[0]

            keys = [
                "num_graphs",
                "dag_num_nodes",
                "original_graph_num_nodes",
                "dag_edge_index",
                "dag_x",
                "dag_layers_mask",
                "dag_readout_at_each_layer",
                "dag_leaves_at_each_layer",
                "edge_attr",
                "edge_multiplicities",
                "dag_edge_attr",
            ]
            values = [
                len(graph_ids),
                canonization_dag_num_nodes,
                batch_data.num_nodes,
                canonization_dag_edge_index,
                dag_x,
                canonization_dag_layers_mask,
                canonization_dag_readout_at_each_layer,
                canonization_dag_leaves_at_each_layer,
                edge_attributes,
                edge_multiplicities,
                dag_edge_attr,
            ]

            for key, value in store.items():
                if key in keys:
                    continue
                if store.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)
            store.update(zip(keys, values))
        data = CanonizedDirectedAcyclicGraphData(**batch_data.to_dict())
        return data
