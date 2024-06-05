from json import dumps, loads

from torch import Tensor, all, any, cat, long, tensor


def find_smallest_missing_integer(numbers: set) -> int:
    """
    Finds the smallest missing integer in the given set of integers.

    :param numbers: The set of integers.
    :return: The smallest missing integer in the given set of integers.
    """
    for i in range(1, max(numbers) + 2):
        if i not in numbers:
            return i


class DAG:
    """
    A class to represent a Directed Acyclic Graph (DAG).
    """

    def __init__(self):
        """
        Initializes the graph as a directed graph.
        """
        self.nodes = set()
        self.leaves = set()
        self.roots = set()
        self.children_map = {}
        self.parent_map = {}

        self.labels = {}
        self.edge_index = Tensor().to(long)

    def get_children(self, node):
        """
        Get the children for the specified node in the DAG tree.

        :param node: The node to get the children for.
        :return: The children for the specified node.
        """
        if node in self.children_map:
            return self.children_map[node]
        elif self.edge_index.numel() == 0:
            return []
        else:
            mask = self.edge_index[1] == node
            successors = self.edge_index[0][mask].unique()
            self.children_map[node] = set(successors.tolist())  # Store children in the map
            return self.children_map[node]

    def get_parents(self, node) -> list:
        """
        Get the parents for the specified node in the DAG tree.

        :param node: The node to get the parents for.
        :return: The parents for the specified node.
        """
        if not isinstance(node, int):
            raise TypeError("The node should be an integer.")
        if node in self.parent_map:
            return self.parent_map[node]
        elif self.edge_index.numel() == 0:
            return []
        else:
            mask = self.edge_index[0] == node
            predecessors = self.edge_index[1][mask].unique()
            self.parent_map[node] = set(predecessors.tolist())  # Store parents in the map
            return self.parent_map[node]

    def add_next_node(self, label):
        """
        Adds a node to the graph.

        :param label: The label for the node that could be redundant
        :return: The ID of the node just added
        """
        node = self.find_free_id()
        self.nodes.add(node)
        self.labels.update({node: label})
        self.leaves.add(node)
        self.roots.add(node)
        self.children_map[node] = set()
        self.parent_map[node] = set()
        return node

    def find_free_id(self):
        """
        Find the smallest positive integer ID that is not already present as a node in the graph.

        :return: int: The smallest positive integer ID that is not present in the graph.
        """
        if self.nodes:
            free_id = find_smallest_missing_integer(self.nodes)
        else:
            free_id = 0
        return free_id

    def add_node(self, node=None, label=None):
        """
        Adds a node to the graph.

        :param node: The node to be added to the DAG.
        :param label: The label for the node that could be redundant
        :return: The ID of the node just added
        """
        label = label if label else node  # this is intentional so label is none for placeholder vertices
        node = node if node else self.find_free_id()
        self.nodes.add(node)
        self.labels.update({node: label})
        self.leaves.add(node)
        self.roots.add(node)
        if node in self.children_map or node in self.parent_map:
            raise ValueError(f"Node {node} already exists in the graph")
        self.children_map[node] = set()
        self.parent_map[node] = set()
        return node

    def add_edge(self, parent, child, allow_redundant_edges=False):
        """
        Adds an edge to the graph from the child to parent node.

        :param parent: The parent node.
        :param child: The child node.
        :param allow_redundant_edges: Whether to allow redundant edges.
        :return: None
        """
        if child in self.get_parents(parent):
            raise ValueError(f"Adding edge from {child} to {parent} will create a cycle in the DAG")

        if not allow_redundant_edges and self.is_edge_exists(parent, child):
            raise ValueError(f"Edge from {child} to {parent} already exists in the DAG")

        self.edge_index = cat([self.edge_index, tensor([[child], [parent]])], dim=1)

        # Update the roots and leaves
        if parent in self.leaves:
            # leaves are all nodes with no incoming edges
            self.leaves.remove(parent)
        if child in self.roots:
            # roots are all nodes with no outgoing edges
            self.roots.remove(child)

        # Update the children and parent maps
        self.children_map[parent].add(child)
        self.parent_map[child].add(parent)

    def is_edge_exists(self, parent, child):
        if self.edge_index.numel() == 0:
            return False
        edge = tensor([child, parent])
        return any(all(self.edge_index.t() == edge, dim=1)).item()

    def get_roots(self):
        """
        Get the root nodes of the DAG.
        :return: A list of ints representing the root nodes.
        """
        if not self.roots:
            return {}
        return self.roots

    def get_label(self, node):
        """
        Retrieve the label of a node in the graph.

        Raises:
        KeyError: If the node does not exist in the graph.

        :param node: The node identifier.
        :return: str: The label of the node.
        """
        return self.labels[node]

    def get_leaves(self):
        """
        Get the leave nodes of the DAG.

        :return: A list of ints representing the leaf nodes.
        """
        # if self.edge_index.numel() == 0: # WHY?????
        if not self.leaves:
            return {}
        return self.leaves

    def __add__(self, dag_j: "DAG") -> "DAG":
        """
        Merge two directed acyclic graphs (DAGs) into a new one.

        :param dag_j: Another DAG to merge with the current DAG.
        :return: The merged DAG.
        """
        if not isinstance(dag_j, DAG):
            raise TypeError(f"Cannot merge DAG with {type(dag_j)}")

        map_dag_j_to_dag_i = {}
        for root in dag_j.roots:
            __add_sub_dag__(self, dag_j, map_dag_j_to_dag_i, root)
        return self

    def __radd__(self, dag_j: "DAG") -> "DAG":
        """
        Merge two directed acyclic graphs (DAGs) into a new one.

        :param dag_j: Another DAG to merge with the current DAG.
        :return: The merged DAG.
        """
        if isinstance(dag_j, DAG):
            return self + dag_j
        return self

    def __len__(self):
        """
        Returns the number of nodes in the DAG.
        """
        return len(self.nodes)

    def __repr__(self):
        """
        Returns a string representation of the DAG.
        """
        return f"DAG(nodes={len(self)}, edges={self.edge_index.numel() // 2})"

    def to_string(self):
        """
        Serializes the object to a JSON-formatted string.

        :return: The JSON-formatted string.
        """
        # Convert sets to lists as sets are not JSON serializable
        data = {
            "nodes": list(self.nodes),
            "leaves": list(self.leaves),
            "roots": list(self.roots),
            "children_map": {k: list(v) for k, v in self.children_map.items()},
            "parent_map": {k: list(v) for k, v in self.parent_map.items()},
            "labels": self.labels,
            "edge_index": self.edge_index.tolist(),
        }
        return dumps(data)

    def from_string(self, s):
        """
        Deserializes the object from a JSON-formatted string.

        :param s: The JSON-formatted string.
        """
        data = loads(s)
        self.nodes = set(data["nodes"])
        self.leaves = set(data["leaves"])
        self.roots = set(data["roots"])
        self.children_map = {k: set(v) for k, v in data["children_map"].items()}
        self.parent_map = {k: set(v) for k, v in data["parent_map"].items()}
        self.labels = data["labels"]
        self.edge_index = Tensor(data["edge_index"]).to(long)


def __add_sub_dag__(dag_i: DAG, dag_j: DAG, map_dag_j_to_dag_i: dict, node: int):
    """
    Adds a sub DAG from `dag_j` to `dag_i`.

    The function maps the nodes of `dag_j` to the corresponding nodes in `dag_i`.
    If a node in `dag_j` already exists in `dag_i`, it is not added again.
    Otherwise, a new node is added to `dag_i`.
    The function updates `map_dag_j_to_dag_i` to reflect the mapping of nodes from `another_dag` to `original_dag`.

    :param dag_i: The DAG to add the sub DAG to.
    :param dag_j: The sub DAG to add to `dag_i`.
    :param map_dag_j_to_dag_i: A dictionary that maps nodes from `dag_j` to nodes in `original_dag`.
    :param node: The node in `dag_j` to add to `dag_i`.
    :return: The updated `dag_i` with the added sub DAG.
    """
    if node in map_dag_j_to_dag_i:
        # node already in DAG
        return dag_i

    dag_j_node_label = dag_j.get_label(node)
    if node in dag_j.get_leaves():
        # find the corresponding leaf in dag and put in a map
        for element in dag_i.get_leaves():
            if dag_i.get_label(element) == dag_j_node_label and len(dag_i.get_children(element)) == 0:
                map_dag_j_to_dag_i[node] = element
        if node not in map_dag_j_to_dag_i:
            # new node has to be inserted
            new_node = dag_i.add_next_node(dag_j_node_label)
            map_dag_j_to_dag_i[node] = new_node
        return dag_i

    dag_j_children_of_node = list(dag_j.get_children(node))
    for child in dag_j_children_of_node:
        # add children first
        __add_sub_dag__(dag_i, dag_j, map_dag_j_to_dag_i, child)

    # find the corresponding vertex in cut of parent sets of children
    possible_matches = {parent for parent in dag_i.get_parents(map_dag_j_to_dag_i[dag_j_children_of_node[0]])}
    for child in dag_j_children_of_node:
        parents = dag_i.get_parents(map_dag_j_to_dag_i[child])
        possible_matches.intersection_update(parents)

    dag_j_children_of_node_len = len(dag_j_children_of_node)
    match = None
    for element in possible_matches:
        has_same_label = dag_i.get_label(element) == dag_j_node_label
        has_same_number_of_children = len(dag_i.get_children(element)) == dag_j_children_of_node_len
        if has_same_label and has_same_number_of_children:
            match, map_dag_j_to_dag_i[node] = element, element
    if match is None:
        # new node has to be inserted
        new_node = dag_i.add_next_node(dag_j_node_label)
        for child in dag_j_children_of_node:
            # add connections to children
            dag_i.add_edge(new_node, map_dag_j_to_dag_i[child])
        map_dag_j_to_dag_i[node] = new_node
    return dag_i


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create an instance of the DAG class
    dag = DAG()

    # --- the nodes need a label it has to be possible to have two 2s for example.
    # --- dag.add_node(identifier, label)
    # Adding nodes to the graph
    dag.add_node(0)
    dag.add_node(1)
    dag.add_node(2)
    dag.add_node(3)
    dag.add_node(4)
    dag.add_node(5)

    # Adding edges to the graph
    dag.add_edge(3, 0)
    dag.add_edge(1, 3)
    dag.add_edge(4, 3)
    dag.add_edge(2, 4)
    dag.add_edge(4, 5)

    # Insert temporary node between 1, and 3
    # --- here the label is missing again
    # dag.insert_node_between(1, 3, 6)

    # Get the parent nodes of a specific node
    print("3's parents: -> ", dag.get_parents(3))

    # Print all roots of the DAG
    print("DAG roots: -> ", dag.get_roots())

    # Print all leaves of the DAG
    print("DAG leaves: -> ", dag.get_leaves())

    # Get the children nodes of a specific node
    print("1's children: -> ", dag.get_children(1))

    from networkx import DiGraph, draw_networkx, from_numpy_array
    from networkx.drawing.nx_agraph import graphviz_layout
    from torch_geometric.utils import to_dense_adj

    dag_networkx = from_numpy_array(to_dense_adj(dag.edge_index)[0].numpy(), create_using=DiGraph)
    pos = graphviz_layout(dag_networkx, prog="dot")
    labels = {n: l for n, l in dag.labels.items()}
    # --- strangely, I had to put _networkx here, otherwise I get a TypeError: '_AxesStack' object is not callable
    draw_networkx(dag_networkx, pos, node_size=900, labels=labels, with_labels=True)
    plt.show()
