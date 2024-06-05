from collections import defaultdict
from multiprocessing import cpu_count
from time import sleep
from typing import Dict, List, Tuple

from joblib import Parallel, delayed
from torch import Tensor, cat, iinfo, int64, isin, long, ones_like, tensor
from tqdm import tqdm

from src.canonized_dag import CanonizedDAG
from src.canonized_ntrees import canonized_n_tree
from src.dag import DAG
from src.ntrees import n_tree

MIN_VALUE = iinfo(int64).min
MIN_NUM_OF_DAGS_TO_ADD_SIMULTANEOUSLY = 2


class UnboundedList(list):
    """
    A list that returns None if the index is out of bounds.
    """

    def __getitem__(self, i: int) -> object | None:
        if len(self) > i:
            return super().__getitem__(i)
        else:
            return None


def compact_n_tree(args) -> DAG:
    """
    Wrapper for n_tree to be used in a multiprocessing pool.
    :param args: The arguments for n_tree.
    :return: The n_tree.
    """
    return n_tree(*args)


def compact_dag_add(args: Tuple[DAG, DAG]) -> DAG:
    """
    Wrapper for DAG add to be used in a multiprocessing pool.
    :param args: The arguments for DAG sum.
    :return: The n_tree.
    """
    dag_i = args[0]
    dag_j = args[1]
    # return empty DAG the two DAGs are None
    if dag_i is None and dag_j is None:
        return DAG()
    # return dag_i if dag_j is None
    if dag_i is None:
        return dag_j
    # return dag_j if dag_i is None
    if dag_j is None:
        return dag_i
    # otherwise, return the sum of the two DAGs
    return dag_i + dag_j


def compact_canonized_n_tree(args) -> Tuple[CanonizedDAG, int]:
    """
    Wrapper for n_tree to be used in a multiprocessing pool.
    :param args: The arguments for n_tree.
    :return: The n_tree.
    """
    return canonized_n_tree(*args), args[1]


def extract_non_canonized_edge_index_per_layer(dag: DAG):
    """
    Sorts the DAG in rooted based order, reverses the graph and extracts the edge indices for each layer.

    :param dag: Directed Acyclic Graph object
    :return: The DAG, the edge indices for each layer and the node labels.
    """
    edge_index_per_layer = []

    current_roots = dag.get_roots()
    processed = set()
    while current_roots:
        new_roots = set()
        for r in current_roots:
            processed.add(r)
        sub_edge_index = Tensor().to(long)
        for node in current_roots:
            for c in dag.get_children(node):
                new_edge = tensor([[c], [node]])
                sub_edge_index = cat((sub_edge_index, new_edge), dim=1)
                if c not in dag.get_roots():
                    pp = True
                    for p in dag.get_parents(c):
                        if p not in processed:
                            pp = False
                    if pp:
                        new_roots.add(c)
        edge_index_per_layer.append(sub_edge_index) if len(sub_edge_index) > 0 else None
        current_roots = new_roots

    return edge_index_per_layer[::-1]


def extract_canonized_edge_index_per_layer(
    dag,
):  # We need the node heights for that. They are only in the canonized DAG
    """
    Extracts the edge indices for each layer.
    :return: The edge indices for each layer
    """
    edge_index_per_layer = dict()
    for node in dag.nodes:
        node_height = dag.get_height(node)
        if node_height > 0:
            for c in dag.get_children(node):
                num_edges = len(dag.get_h_edge_attribute(node, c))
                new_edge = tensor(
                    [
                        [
                            c,
                        ]
                        * num_edges,
                        [
                            node,
                        ]
                        * num_edges,
                    ]
                )
                edge_index_per_layer[node_height] = (
                    cat((edge_index_per_layer[node_height], new_edge), dim=1)
                    if node_height in edge_index_per_layer
                    else new_edge
                )
    return list(edge_index_per_layer.values())


class TensorDAG:
    """
    Computes the DAG for a given graph and returns the masks for each layer.
    """

    def __init__(self, number_of_nodes, number_of_layer: int, k: int = None, n_jobs: int = -1, verbose: bool = True):
        """
        :param number_of_nodes: The number of nodes in the graph.
        :param number_of_layer: The number of layers in the DAG.
        :param k: The redundancy parameter. A node can occur up to k layers after their first occurrence.
        :param n_jobs: The number of processes to use for parallelization. -1 means all available processes.
        :param verbose: Print progress bar.
        """
        self.number_of_nodes: int = number_of_nodes
        self.number_of_layer: int = number_of_layer
        self.k: int = self.number_of_layer if k is None else k
        self.n_jobs: int = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self.verbose: bool = verbose
        self.is_built: bool = False
        self.dag: DAG = None
        self.edge_indices_per_layer: List[Tensor] = None
        self.labels: dict = None

    def get_dag_edge_index(self) -> Tensor:
        """
        Returns the edge indices of the computed DAG.

        :return: The edge indices of the computed DAG.
        """
        if not self.is_built:
            raise Exception("The DAG is not built yet.")
        return self.dag.edge_index

    def get_dag_leaves(self) -> Tensor:
        """
        Returns the leaves of the computed DAG.

        :return: The leaves of the computed DAG.
        """
        if not self.is_built:
            raise Exception("The DAG is not built yet.")
        return self.dag.get_leaves()

    def get_dag_roots(self) -> Tensor:
        """
        Returns the roots of the computed DAG.

        :return: The roots of the computed DAG.
        """
        if not self.is_built:
            raise Exception("The DAG is not built yet.")
        return self.dag.get_roots()

    def get_dag(self) -> DAG | CanonizedDAG:
        """
        Returns the DAG.

        :return: The DAG.
        """
        if not self.is_built:
            raise Exception("The DAG is not built yet.")
        return self.dag

    def __call__(self, edge_index: Tensor) -> None:
        """
        Builds the DAG for the given graph.

        :param edge_index: The edge index of the graph.
        :return: The DAG for the given graph.
        """
        node_ids = edge_index.unique().tolist()
        self.number_of_layer = self.number_of_layer + 1
        if node_ids:
            args_list = [(edge_index, node_i, self.number_of_layer, self.k) for node_i in node_ids]
            args_list = tqdm(args_list, disable=not self.verbose)
            # no parallelization
            if self.n_jobs == 1:
                dag = sum([*map(compact_n_tree, args_list)])
            else:
                # parallelize the creation of the n-trees
                results = Parallel(n_jobs=self.n_jobs)(delayed(compact_n_tree)(args) for args in args_list)
                results = UnboundedList(results)
                # parallelize the addition of the n-trees
                while len(results) > MIN_NUM_OF_DAGS_TO_ADD_SIMULTANEOUSLY:
                    # construct the n-trees in pairs
                    n_tree_pairs = UnboundedList([(results[i], results[i + 1]) for i in range(0, len(results), 2)])
                    results = Parallel(n_jobs=self.n_jobs)(
                        delayed(compact_dag_add)(args) for args in tqdm(n_tree_pairs, disable=not self.verbose)
                    )
                    results = UnboundedList(results)
                dag_sum = DAG()
                for dag in tqdm(results, disable=not self.verbose):
                    dag_sum = dag_sum + dag
                dag = dag_sum
        else:
            dag = DAG()

        self.dag = dag
        self.labels = list(map(int, dag.labels.values()))
        self.edge_indices_per_layer = extract_non_canonized_edge_index_per_layer(dag)
        self.is_built = True
        return dag.edge_index

    def compute_masked_indices(self, dag_main_edge_index) -> Tensor:
        """
        Computes the masked indices for each layer.

        :param dag_main_edge_index:  The edge index of the graph.
        :return: The padded masked indices for each layer.
        """
        if dag_main_edge_index is None and not self.is_built:
            raise Exception("Either the DAG is not built yet or the graph edge index is not provided.")
        if dag_main_edge_index is None:
            dag_main_edge_index = self.get_dag_edge_index()

        layer_masks = ones_like(dag_main_edge_index[0]) * -1
        for i, dag_edge_index in tqdm(enumerate(self.edge_indices_per_layer), disable=not self.verbose):
            mask_0 = isin(dag_main_edge_index[0], dag_edge_index[0]).nonzero().squeeze(-1).tolist()
            mask_1 = isin(dag_main_edge_index[1], dag_edge_index[1]).nonzero().squeeze(-1).tolist()
            mask = list(set(mask_0).intersection(set(mask_1)))
            layer_masks[mask] = i
        return layer_masks


def build_canonized_n_tree_per_layer_for_each_node(
    edge_index: Tensor,
    node_attributes: Tensor,
    edge_attributes: Tensor,
    num_nodes: int,
    number_of_layer: int,
    k: int,
    n_jobs: int,
    verbose: bool,
) -> List[List[Tuple[CanonizedDAG, int]]]:
    """
    Builds the DAG for the given graph for each node and each layer.

    :param edge_index: The edge index of the graph.
    :param node_attributes: The node attributes of the graph.
    :param edge_attributes: The edge attributes of the graph.
    :param num_nodes: The number of nodes in the graph.
    :param number_of_layer: The number of layers in the DAG.
    :param k: The redundancy parameter. A node can occur up to k layers after their first occurrence.
    :param n_jobs: The number of processes to use for parallelization. -1 means all available processes.
    :param verbose: Print progress bar.
    :return: The DAG for the given graph for each node and each layer.
    """
    dags_and_node_ids_at_different_layers = []
    edge_attributes = (
        {(i, j): 1.0 for i, j in edge_index.t().tolist()}
        if edge_attributes is None
        else {(i, j): e for (i, j), e in zip(*[edge_index.t().tolist(), edge_attributes])}
    )
    for num_layer in range(number_of_layer):
        # construct the n-trees arguments for non-isolated nodes
        nodes_args_list = [
            (edge_index, node_i, node_attributes, edge_attributes, num_layer + 1, k) for node_i in range(num_nodes)
        ]
        nodes_args_list = tqdm(nodes_args_list, disable=not verbose, desc=f"Layer {num_layer}/{number_of_layer - 1}")
        if n_jobs == 1:
            dags_and_node_ids = [*map(compact_canonized_n_tree, nodes_args_list)]
        else:
            # parallelize the creation of the n-trees for nodes
            dags_and_node_ids = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(compact_canonized_n_tree)(args) for args in nodes_args_list
            )
            # wait until all the n-trees are created
            while len(dags_and_node_ids) < num_nodes:
                sleep(1.0)

        # add the n-trees to the list of n-trees at different layers
        dags_and_node_ids_at_different_layers.append(dags_and_node_ids)
    return dags_and_node_ids_at_different_layers


def extract_canonization_for_leaves_and_roots(
    dag_and_node_ids: List[Tuple[CanonizedDAG, int]]
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    Extracts the canonization of the leaves and the roots for each node.

    :param dag_and_node_ids: The DAGs for each node.
    :return: The DAG, the canonization of the leaves and the roots for each node.
    """
    # extract the canonization of the roots for each node and each layer
    node_to_root_canonization = {
        node_i: dag_i.get_canonization(list(dag_i.get_roots())[0]) for dag_i, node_i in dag_and_node_ids
    }

    # extract the canonization of the leaves for each node and each layer
    node_to_leave_canonization = defaultdict(list)
    _ = {
        node_to_leave_canonization[node_i].append(dag_i.get_canonization(leaf))
        for dag_i, node_i in dag_and_node_ids
        for leaf in dag_i.get_leaves()
    }

    return node_to_root_canonization, node_to_leave_canonization


def extract_leaves_and_readout_from_canonization(
    dag: CanonizedDAG,
    node_to_root_canonization_at_each_layer: List[List[int]],
    node_to_leave_canonization_at_each_layer: List[List[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extracts the leaves and the readout from the canonization of the leaves and the roots for each node.

    :param dag: The canonized DAG.
    :param node_to_root_canonization_at_each_layer: The canonization of the roots for each node.
    :param node_to_leave_canonization_at_each_layer: The canonization of the leaves for each node.
    :return: The leaves and the readout for each node.
    """
    readout_at_each_layer: List[List[int]] = []
    leaves_at_each_layer: List[List[int]] = []

    # extract the canonization of the leaves and the roots for each node and each layer
    for node_to_leave_canonization_at_layer_i in node_to_leave_canonization_at_each_layer:
        unsorted_node_to_leave_at_layer_i = {
            node_i: dag.canonization_map[canonization_i_j]
            for node_i, canonization_i in node_to_leave_canonization_at_layer_i.items()
            for canonization_i_j in canonization_i
        }
        sorted_node_to_leave_at_layer_i = sorted(unsorted_node_to_leave_at_layer_i.items(), key=lambda x: x[0])
        leaves_at_each_layer += [[*zip(*sorted_node_to_leave_at_layer_i)][1]]
    # extract the canonization of the leaves and the roots for each node and each layer
    for node_to_root_canonization_at_layer_i in node_to_root_canonization_at_each_layer:
        unsorted_node_to_readout_at_layer_i = {
            node_i: dag.canonization_map[canonization_i]
            for node_i, canonization_i in node_to_root_canonization_at_layer_i.items()
        }
        sorted_node_to_readout_at_layer_i = sorted(unsorted_node_to_readout_at_layer_i.items(), key=lambda x: x[0])
        readout_at_each_layer += [[*zip(*sorted_node_to_readout_at_layer_i)][1]]

    return leaves_at_each_layer, readout_at_each_layer


def extract_leaves_and_roots_canonization_per_layer(
    dags_and_node_ids_at_each_layers: List[List[Tuple[CanonizedDAG, int]]]
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Sums the DAGs for each node and each layer and extracts the canonization of the leaves and the roots
    for each node and each layer.

    :param dags_and_node_ids_at_each_layers: The DAGs for each node and each layer.
    :return: The DAG, the canonization of the leaves and the roots for each node and each layer.
    """
    node_to_root_canonization_at_each_layer, node_to_leave_canonization_at_each_layer = [], []
    for dag_and_node_ids_at_layer_i in dags_and_node_ids_at_each_layers:
        # extract the canonization of the leaves and the roots for each node and each layer
        (
            node_to_root_canonization_at_layer_i,
            node_to_leave_canonization_at_layer_i,
        ) = extract_canonization_for_leaves_and_roots(dag_and_node_ids_at_layer_i)
        node_to_root_canonization_at_each_layer += [node_to_root_canonization_at_layer_i]
        node_to_leave_canonization_at_each_layer += [node_to_leave_canonization_at_layer_i]
    return node_to_root_canonization_at_each_layer, node_to_leave_canonization_at_each_layer


class TensorCanonizedDAG(TensorDAG):
    def __init__(self, number_of_nodes, number_of_layer: int, k: int = None, n_jobs: int = -1, verbose: bool = True):
        super().__init__(number_of_nodes, number_of_layer, k, n_jobs, verbose)
        self.readout_at_each_layer: List[List[int]] = []
        self.leaves_at_each_layer: List[List[int]] = []

    def __call__(self, edge_index: Tensor, node_attributes: Tensor, edge_attributes: Tensor) -> None:
        """
        Builds the DAG for the given graph.

        :param edge_index: The edge index of the graph.
        :param node_attributes: The node attributes of the graph.
        :param edge_attributes: The edge attributes of the graph.
        :return: The DAG for the given graph.
        """
        num_nodes = (
            max([self.number_of_nodes, node_attributes.shape[0]]) if self.number_of_nodes else node_attributes.shape[0]
        )
        dags_and_node_ids_at_different_layers = build_canonized_n_tree_per_layer_for_each_node(
            edge_index,
            node_attributes,
            edge_attributes,
            num_nodes,
            self.number_of_layer,
            self.k,
            self.n_jobs,
            self.verbose,
        )
        # Extract the canonization of the leaves and the roots for each node, and each layer
        # Important: The order of the nodes is important for the canonization,
        # so this step should be done before summing the DAGs
        (
            node_to_root_canonization_at_each_layer,
            node_to_leave_canonization_at_each_layer,
        ) = extract_leaves_and_roots_canonization_per_layer(dags_and_node_ids_at_different_layers)
        if self.verbose:
            print("Summing the Canonized DAGs for each node and each layer.")
        # Sum the DAGs for each node and each layer
        if self.n_jobs == 1:
            dags_at_each_layer = [*map(lambda arr: [*zip(*arr)][0], dags_and_node_ids_at_different_layers)]
            dag = sum([*map(sum, dags_at_each_layer)])
        else:
            dag_sum = CanonizedDAG()
            for dags_at_each_layer in tqdm(
                [*map(lambda arr: [*zip(*arr)][0], dags_and_node_ids_at_different_layers)], disable=not self.verbose
            ):
                # parallelize the addition of the n-trees
                dags_at_each_layer = UnboundedList(dags_at_each_layer)
                while len(dags_at_each_layer) > MIN_NUM_OF_DAGS_TO_ADD_SIMULTANEOUSLY:
                    # construct the n-trees in pairs
                    n_tree_pairs = UnboundedList(
                        [
                            (dags_at_each_layer[i], dags_at_each_layer[i + 1])
                            for i in range(0, len(dags_at_each_layer), 2)
                        ]
                    )
                    dags_at_each_layer = Parallel(n_jobs=self.n_jobs)(
                        delayed(compact_dag_add)(args) for args in tqdm(n_tree_pairs, disable=not self.verbose)
                    )
                    dags_at_each_layer = UnboundedList(dags_at_each_layer)
                for dag_i in tqdm(dags_at_each_layer, disable=not self.verbose):
                    dag_sum = dag_sum + dag_i
            dag = dag_sum
        # Extract the canonization of the leaves and the roots for each node, and each layer
        self.leaves_at_each_layer, self.readout_at_each_layer = extract_leaves_and_readout_from_canonization(
            dag, node_to_root_canonization_at_each_layer, node_to_leave_canonization_at_each_layer
        )
        self.edge_indices_per_layer = extract_canonized_edge_index_per_layer(dag)
        self.dag = dag
        self.is_built = True
        return dag.edge_index


def process_graphs_canonized_n_tree(args):
    """
    Builds the DAG for the given graph for each node and each layer.

    :param graph_i: The index of the graph.
    :param original_edge_indices: The edge indices of the graphs.
    :param original_nodes_attributes: The node attributes of the graphs.
    :param original_edge_attributes: The edge attributes of the graphs.
    :param number_of_nodes: The number of nodes in the graph.
    :param number_of_layer: The number of layers in the DAG.
    :param k: The redundancy parameter. A node can occur up to k layers after their first occurrence.
    :param n_jobs: The number of processes to use for parallelization. -1 means all available processes.
    :return: The DAG for the given graph for each node and each layer.
    """
    (
        graph_i,
        original_edge_indices,
        original_nodes_attributes,
        original_edge_attributes,
        number_of_nodes,
        number_of_layer,
        k,
        n_jobs,
    ) = args
    graph_i_edge_index = original_edge_indices[graph_i]
    graph_i_node_attributes = original_nodes_attributes[graph_i]
    graph_i_edge_attributes = original_edge_attributes[graph_i]
    graph_i_num_nodes = (
        max([number_of_nodes, graph_i_node_attributes.shape[0]])
        if number_of_nodes
        else graph_i_node_attributes.shape[0]
    )
    dags_and_node_ids_at_different_layers = build_canonized_n_tree_per_layer_for_each_node(
        graph_i_edge_index,
        graph_i_node_attributes,
        graph_i_edge_attributes,
        graph_i_num_nodes,
        number_of_layer,
        k,
        n_jobs,
        False,
    )
    (
        graph_i_node_to_root_canonization_at_each_layer,
        graph_i_node_to_leave_canonization_at_each_layer,
    ) = extract_leaves_and_roots_canonization_per_layer(dags_and_node_ids_at_different_layers)
    dags_at_each_layer = [*map(lambda arr: [*zip(*arr)][0], dags_and_node_ids_at_different_layers)]
    dag = sum([*map(sum, dags_at_each_layer)])
    return dag, graph_i_node_to_root_canonization_at_each_layer, graph_i_node_to_leave_canonization_at_each_layer


def process_canonization_dag_terminals(
    graph_i, dag, node_to_root_canonization_at_each_layer_per_graph, node_to_leave_canonization_at_each_layer_per_graph
):
    """
    Extracts the canonization of the leaves and the roots for each node.

    :param graph_i: The index of the graph.
    :param dag: The DAG.
    :param node_to_root_canonization_at_each_layer_per_graph: The canonization of the roots for each node.
    :param node_to_leave_canonization_at_each_layer_per_graph: The canonization of the leaves for each node.
    :return: The DAG, the canonization of the leaves and the roots for each node.
    """
    graph_i_node_to_root_canonization_at_each_layer = node_to_root_canonization_at_each_layer_per_graph[graph_i]
    graph_i_node_to_leave_canonization_at_each_layer = node_to_leave_canonization_at_each_layer_per_graph[graph_i]

    leaves_at_each_layer, readout_at_each_layer = extract_leaves_and_readout_from_canonization(
        dag, graph_i_node_to_root_canonization_at_each_layer, graph_i_node_to_leave_canonization_at_each_layer
    )
    return leaves_at_each_layer, readout_at_each_layer


def add_graphs(
    original_edge_indices: List[Tensor],
    original_nodes_attributes: List[Tensor],
    original_edge_attributes: List[Tensor],
    number_of_nodes: int,
    number_of_layer: int,
    k: int,
    n_jobs_graphs: int,
    n_jobs_dag: int,
    verbose: bool,
) -> Tuple[List[CanonizedDAG], List[List[dict]], List[List[dict]]]:
    """
    Builds the DAG for the given graph for each node and each layer and sums the DAGs for each graph.

    :param original_edge_indices: The edge indices of the graphs.
    :param original_nodes_attributes: The node attributes of the graphs.
    :param original_edge_attributes: The edge attributes of the graphs.
    :param number_of_nodes: The number of nodes in the graph.
    :param number_of_layer: The number of layers in the DAG.
    :param k: The redundancy parameter, node can occur up to k layers after their first occurrence.
    :param n_jobs_graphs: The number of processes to use for parallelization. -1 means all available processes.
    :param n_jobs_dag: The number of processes to use for parallelization. -1 means all available processes.
    :param verbose: Print progress bar.
    :return: The DAG for the given graph for each node and each layer.
    """

    assert len(original_edge_indices) == len(
        original_nodes_attributes
    ), "The number of edge indices and node attributes should be the same."
    batch_number_of_graphs = len(original_edge_indices)

    args_list = [
        (
            graph_i,
            original_edge_indices,
            original_nodes_attributes,
            original_edge_attributes,
            number_of_nodes,
            number_of_layer,
            k,
            n_jobs_dag,
        )
        for graph_i in range(batch_number_of_graphs)
    ]
    args_list = tqdm(args_list, disable=not verbose)
    if n_jobs_graphs == 1:
        processed_graphs_canonized_n_tree_results = [*map(process_graphs_canonized_n_tree, args_list)]
    else:
        processed_graphs_canonized_n_tree_results = Parallel(n_jobs=n_jobs_graphs, backend="threading")(
            delayed(process_graphs_canonized_n_tree)(args_list_i) for args_list_i in args_list
        )

    # wait until all the n-trees are created
    while len(processed_graphs_canonized_n_tree_results) < batch_number_of_graphs:
        sleep(1.0)

    dags_per_graph: List[Tuple[CanonizedDAG, int]] = [item[0] for item in processed_graphs_canonized_n_tree_results]
    node_to_root_canonization_at_each_layer_per_graph: Dict[int, int] = [
        item[1] for item in processed_graphs_canonized_n_tree_results
    ]
    node_to_leave_canonization_at_each_layer_per_graph: Dict[int, List[int]] = [
        item[2] for item in processed_graphs_canonized_n_tree_results
    ]
    return (
        dags_per_graph,
        node_to_root_canonization_at_each_layer_per_graph,
        node_to_leave_canonization_at_each_layer_per_graph,
    )


class BatchTensorCanonizedDAG(TensorCanonizedDAG):
    def __init__(
        self, number_of_layer: int, k: int = None, n_jobs_graphs: int = -1, n_jobs_dag: int = -1, verbose: bool = True
    ):
        super().__init__(None, number_of_layer, k, n_jobs_dag, verbose)
        self.n_jobs_graphs: int = cpu_count() if n_jobs_graphs == -1 else max(1, n_jobs_graphs)
        self.n_jobs_dag: int = cpu_count() if n_jobs_dag == -1 else max(1, n_jobs_dag)
        self.readout_at_each_layer: List[List[int]] = []
        self.leaves_at_each_layer: List[List[int]] = []

    def __call__(
        self,
        dags_per_graph: List[CanonizedDAG],
        node_to_root_canonization_at_each_layer_per_graph: List[List[dict]],
        node_to_leave_canonization_at_each_layer_per_graph: List[List[dict]],
    ) -> None:
        """
        Computes the DAG for the given graph and returns the masks for each layer.

        :param dags_per_graph: The DAGs for each graph.
        :param node_to_root_canonization_at_each_layer_per_graph: The canonization of the roots for each node.
        :param node_to_leave_canonization_at_each_layer_per_graph: The canonization of the leaves for each node.
        :return: The masks for each layer.
        """
        batch_number_of_graphs = len(dags_per_graph)
        dag = sum(dags_per_graph)

        if self.n_jobs_graphs == 1:
            processed_canonization_dag_terminals_results = [
                process_canonization_dag_terminals(
                    graph_i,
                    dag,
                    node_to_root_canonization_at_each_layer_per_graph,
                    node_to_leave_canonization_at_each_layer_per_graph,
                )
                for graph_i in tqdm(range(batch_number_of_graphs), disable=not self.verbose)
            ]
        else:
            processed_canonization_dag_terminals_results = Parallel(n_jobs=self.n_jobs_graphs, backend="threading")(
                delayed(process_canonization_dag_terminals)(
                    graph_i,
                    dag,
                    node_to_root_canonization_at_each_layer_per_graph,
                    node_to_leave_canonization_at_each_layer_per_graph,
                )
                for graph_i in tqdm(range(batch_number_of_graphs), disable=not self.verbose)
            )

        # wait until all the n-trees are created
        while len(processed_canonization_dag_terminals_results) < batch_number_of_graphs:
            sleep(1.0)

        leaves_at_each_layer_per_graph = [item[0] for item in processed_canonization_dag_terminals_results]
        readout_at_each_layer_per_graph = [item[1] for item in processed_canonization_dag_terminals_results]

        self.leaves_at_each_layer = leaves_at_each_layer_per_graph
        self.readout_at_each_layer = readout_at_each_layer_per_graph
        self.edge_indices_per_layer = extract_canonized_edge_index_per_layer(dag)
        self.dag = dag
        self.is_built = True
        return dag.edge_index
