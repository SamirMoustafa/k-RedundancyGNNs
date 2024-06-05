from collections import deque
from itertools import repeat
from numbers import Number
from typing import Dict

from torch import Tensor

from src.canonized_dag import CanonizedDAG


def in_neighbors(args):
    """
    Returns the in-neighbors of node n_i.
    """
    n_i, edge_index = args
    return set(edge_index[1][edge_index[0] == n_i].tolist())


def out_neighbors(args):
    """
    Returns the out-neighbors of node n_i.
    """
    n_i, edge_index = args
    return set(edge_index[0][edge_index[1] == n_i].tolist())


def empty_canonized_n_tree(vertex: int, h_node_attributes: Tensor) -> CanonizedDAG:
    """
    Generates an empty neighborhood tree (canonized DAG) of vertex.

    :param vertex: The node that will be the root of the neighborhood tree.
    :param h_node_attributes: The (hashed values) of the node attributes.
    :return: Neighborhood tree
    """
    tree = CanonizedDAG()
    tree.add_node(h_node_attributes[vertex], 0)  # root
    tree.canonize()
    tree.fill_canonization_map()
    return tree


def canonized_n_tree(
    edge_index: Tensor,
    vertex: int,
    h_node_attributes: Tensor,
    h_edge_attributes: Dict[tuple, Number],
    height: int,
    k: int,
    flow: str = "source_to_target",
) -> CanonizedDAG:
    """
    Generates a neighborhood tree (canonized DAG) of vertex.

    :param edge_index: The edge index tensor of the underlying graph.
    :param vertex: The node that will be the root of the neighborhood tree.
    :param h_node_attributes: The (hashed values) of the node attributes.
    :param h_edge_attributes: The (hashed values) of the edge attributes.
    :param height: The maximum height of the tree.
    :param k: Redundancy parameter, node can occur up to k layers after their first occurrence.
    :param flow: The flow of the edges, either "source_to_target" or "target_to_source".
    :return: Neighborhood tree
    """
    if flow not in ["source_to_target", "target_to_source"]:
        raise ValueError(f"Expected 'flow' to be either 'source_to_target' or 'target_to_source' (got '{flow}')")

    # Check if vertex has any edges
    is_not_in_edges = not (vertex == edge_index[0]).any() and not (vertex == edge_index[1]).any()
    if is_not_in_edges:
        return empty_canonized_n_tree(vertex, h_node_attributes)

    tree: CanonizedDAG = CanonizedDAG()
    current_id: int = 0
    depth_dict: dict = {vertex: 0}
    leaves: set = {current_id}
    map_id_to_node: dict = {current_id: vertex}
    tree.add_node(h_node_attributes[vertex], current_id)  # root
    current_id += 1
    is_source_to_target = flow == "source_to_target"

    # Store the neighbors of each node in a dictionary
    src_nodes = set(edge_index[0].tolist())
    neighbors_function = in_neighbors if is_source_to_target else out_neighbors
    neighbors_function_args = [*zip(src_nodes, repeat(edge_index))]
    neighbors_dict = dict([*zip(edge_index.unique().tolist(), repeat(set()))])
    neighbors_dict.update(dict(zip(src_nodes, list(map(neighbors_function, neighbors_function_args)))))

    for h in range(1, height):
        new_leaves: set = set()
        found_dict: dict = {}
        leaves_q: deque = deque(leaves)
        while len(leaves_q) > 0:
            current: int = leaves_q.popleft()
            current_node: int = map_id_to_node[current]
            for n in neighbors_dict[current_node]:
                if n not in depth_dict:  # that have not been found previously
                    depth_dict[n] = h
                if depth_dict[n] + k >= h:  # or have only been found this iteration
                    if n not in found_dict:
                        found_dict[n] = current_id
                        tree.add_node(h_node_attributes[n], current_id)  # add node to tree
                        new_leaves.add(current_id)
                        map_id_to_node[current_id] = n
                        current_id += 1
                    h_edge_attribute_idx = (current_node, n) if is_source_to_target else (n, current_node)
                    tree.add_edge(
                        parent=current,
                        child=found_dict[n],
                        h_edge_attribute=h_edge_attributes[h_edge_attribute_idx],
                        edge_multiplicity=1,
                    )
        leaves = new_leaves
    tree.canonize()
    tree.fill_canonization_map()
    return tree
