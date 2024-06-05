from collections import deque

from torch import Tensor

from src.dag import DAG


def n_tree(edge_index: Tensor, vertex: int, height: int, k: int = 0) -> DAG:
    """
    Generates a neighborhood tree (DAG) of vertex.

    :param edge_index: The edge index tensor of the underlying graph.
    :param vertex: The node that will be the root of the neighborhood tree.
    :param height: The maximum height of the tree.
    :param k: Redundancy parameter.
    A node can occur up to k layers after their first occurrence.
    :return: Neighborhood tree
    """
    n = edge_index.max().item() if edge_index.numel() > 0 else 0

    # Check if vertex is in the graph
    is_outside_graph = vertex < 0 or vertex > n
    is_not_in_edges = not (vertex == edge_index[0]).any() and not (vertex == edge_index[1]).any()
    if is_outside_graph or is_not_in_edges:
        return DAG()

    tree: DAG = DAG()
    current_id: int = 0
    depth_dict: dict = {vertex: 0}
    leaves: set = {current_id}
    map_id_to_node: dict = {current_id: vertex}
    tree.add_node(current_id, str(vertex))  # root
    current_id += 1

    # Store the neighbors of each node in a dictionary
    src_nodes = set(edge_index[0].tolist())
    neighbor_func = lambda n_i: set(edge_index[1, Tensor(edge_index[0, :] == n_i).nonzero(as_tuple=True)[0]].tolist())
    neighbor_list = list(map(neighbor_func, src_nodes))
    neighbor_dict = dict(zip(src_nodes, neighbor_list))

    for h in range(1, height):
        new_leaves: set = set()
        found_dict: dict = {}
        leaves_q: deque = deque(leaves)
        while len(leaves_q) > 0:
            current: int = leaves_q.popleft()
            current_node: int = map_id_to_node[current]
            neighbors: set = neighbor_dict[current_node] if current_node in neighbor_dict else set()
            for n in neighbors:
                if n not in depth_dict:  # that have not been found previously
                    depth_dict[n] = h
                if depth_dict[n] + k >= h:  # or have only been found this iteration
                    if n not in found_dict:
                        found_dict[n] = current_id
                        tree.add_node(current_id, str(n))  # add node to tree
                        new_leaves.add(current_id)
                        map_id_to_node[current_id] = n
                        current_id += 1
                    tree.add_edge(current, found_dict[n])  # add edge to tree
        leaves = new_leaves
    return tree
