from collections import defaultdict

from torch import Tensor, long

from src import DAG
from src.hash_function import hash


def get_sorted_string(items):
    """
    Sorts items lexicographically and returns them in a String
    :param items: Strings to be sorted
    """
    items.sort()
    string = ""
    for item in items:
        string += item
    return string


class CanonizedDAG(DAG):
    """
    A class to represent a Directed Acyclic Graph (DAG) that can be canonized.
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
        self.canonization = {}
        self.h_node_attributes = {}
        self.h_edge_attributes = defaultdict(list)
        self.edge_multiplicities = defaultdict(dict)
        self.node_heights = None
        self.edge_index = Tensor().to(long)
        self.canonization_map = dict()

    def add_next_node(self, h_node_attribute):
        """
        Adds a node to the graph.

        :param h_node_attribute: The (hashed value) of the node attribute.
        :return: The ID of the node just added
        """
        node = self.find_free_id()
        self.nodes.add(node)
        self.h_node_attributes.update({node: hash(h_node_attribute)})
        self.leaves.add(node)
        self.roots.add(node)
        self.children_map[node] = set()
        self.parent_map[node] = set()
        return node

    def add_node(self, h_node_attribute, node=None):
        """
        Adds a node to the graph.
        :param h_node_attribute: The (hashed value) of the node attribute.
        :param node: The node to be added to the DAG.
        :return: None
        """
        node = node if node else self.find_free_id()
        self.nodes.add(node)
        self.h_node_attributes.update({node: hash(h_node_attribute)})
        self.leaves.add(node)
        self.roots.add(node)
        if node in self.children_map or node in self.parent_map:
            raise ValueError(f"Node {node} already exists in the graph")
        self.children_map[node] = set()
        self.parent_map[node] = set()
        return node

    def set_edge_multiplicity(self, parent, child, h_edge_attribute, edge_multiplicity):
        """
        Sets the edge multiplicity of the edge.

        :param parent: The parent node.
        :param child: The child node.
        :param h_edge_attribute: The (hashed value) of the edge attribute.
        :param edge_multiplicity: The edge multiplicity.
        """
        self.edge_multiplicities[(child, parent)][h_edge_attribute] = edge_multiplicity

    def add_edge(self, parent, child, h_edge_attribute, edge_multiplicity):
        """
        Adds an edge to the graph from the child to parent node.

        :param parent: The parent node.
        :param child: The child node.
        :param h_edge_attribute: The (hashed value) of the edge attribute.
        :param edge_multiplicity: The edge multiplicity.
        :return: None
        """
        super().add_edge(parent, child, allow_redundant_edges=True)
        self.h_edge_attributes[(child, parent)].append(hash(h_edge_attribute))
        self.edge_multiplicities[(child, parent)][hash(h_edge_attribute)] = edge_multiplicity

    def canonize(self):
        """
        Canonizes the DAG nodes (stores the canonization for the subtree rooted at each node).
        TODO: I am not sure whether we need to make sure, that the DAG is not changed after canonization
            (because then the canonization changes) or whether we need to check if canonization is already computed
        """
        if self.node_heights is None:
            self.compute_heights()
        for root in self.roots:
            self.canonize_node(root)

    def canonize_node(self, node):
        """
        Computes the canonization for the node (and its subtree if necessary).
        :param node: The node to be canonized.
        """
        child_cans = list()
        # canonization is node_attribute[(edge_attribute_1,child_1)...(edge_attribute_n,child_n)],
        # where the children are sorted lexicographically and child_i corresponds to the canonization
        for child in self.get_children(node):
            h_edge_attribute = "-".join(self.h_edge_attributes.get((child, node)))
            child_cans.append("(" + str(h_edge_attribute) + "," + self.get_canonization(child) + ")")
        can_str = ""
        can_str += str(self.h_node_attributes.get(node))
        can_str += "[" + get_sorted_string(child_cans) + "]"
        can_str = hash(can_str)
        self.canonization[node] = can_str

    def get_h_node_attribute(self, node):
        return self.h_node_attributes[node]

    def get_h_edge_attribute(self, parent, child):
        return self.h_edge_attributes[(child, parent)]

    def get_canonization(self, node):
        """
        Returns the canonization of the given node (computes it if necessary).
        :param node: node for that the canonization will be returned
        """
        if node not in self.canonization:
            self.canonize_node(node)
        return self.canonization.get(node)

    def fill_canonization_map(self):
        """
        Fills the map canonization -> node (needed for adding multiple NTrees of different heights into single DAG).
          Should only be used if NTree is already compressed (so canonization -> node is unique)
        """
        for root in self.roots:
            self.fill_canonization_map_node(root)

    def fill_canonization_map_node(self, node):
        """
        Fills the map canonization -> node (needed for adding multiple NTrees of different heights into single DAG).
        :param node: root of subDAG for that the canonization will be added
        """
        if self.get_canonization(node) not in self.canonization_map:
            self.canonization_map[self.get_canonization(node)] = node
            for child in self.get_children(node):
                self.fill_canonization_map_node(child)

    def get_canonization_map(self):
        return self.canonization_map

    def compute_heights(self):
        """
        Computes the heights of the subDAGs in DAG.
        """
        self.node_heights = {}
        for root in self.get_roots():
            self.compute_height(root)

    def compute_height(self, node):
        """
        Computes the height of the node in DAG.
        """
        if node in self.node_heights:
            return
        if node in self.get_leaves():
            self.node_heights[node] = 0
        max_height = 0
        for child in self.get_children(node):
            if child not in self.node_heights:
                self.compute_height(child)
            if self.node_heights[child] > max_height:
                max_height = self.node_heights[child]
        self.node_heights[node] = max_height + 1

    def get_height(self, node):
        """
        Return the height of the given node. The height is the height of the highest subDAG of node.
        :param node: The node to get its height in the DAG.
        :return: The height of the node.
        """
        if self.node_heights is None:
            self.compute_heights()
        return self.node_heights[node]

    def clear_heights(self):
        """
        Clears the node heights, if they should be computed again
        """
        self.node_heights = None

    def __add__(self, dag_j: "CanonizedDAG") -> "CanonizedDAG":
        """
        Merge two directed acyclic graphs (DAGs) into a new one.

        :param dag_j: Another DAG to merge with the current DAG.
        :return: The merged DAG.
        """
        if not len(self):  # dag is empty, so we might have to compress dag_j
            if not len(dag_j):  # both dags are empty
                return dag_j
            self = CanonizedDAG()
            self.add_node(dag_j.get_h_node_attribute(list(dag_j.get_leaves())[0]))
            self.canonize()
            self.fill_canonization_map()

        map_dag_j_to_dag_i = {}
        for root in dag_j.roots:
            __add_sub_dag__(self, dag_j, map_dag_j_to_dag_i, root)
        return self

    def __repr__(self):
        """
        Returns a string representation of the canonized DAG. #TODO: Doesnt this only return the number of nodes/edges?
        """
        return f"CanonizedDAG(nodes={len(self)}, edges={self.edge_index.numel() // 2})"


def __add_sub_dag__(dag_i: CanonizedDAG, dag_j: CanonizedDAG, map_dag_j_to_dag_i: dict, node: int):
    """
    Adds two DAGs.
    The function maps the nodes of `dag_j` to the corresponding nodes in `dag_i`.
    If a node in `dag_j` already exists in `dag_i`, it is not added again.
    Otherwise, a new node is added to `dag_i`.
    The function updates `map_dag_j_to_dag_i` to reflect the mapping of nodes from `dag_j` to `dag_i`.
    :param dag_i: The DAG to add `dag_j` to.
    :param dag_j: The sub DAG to add to `dag_i`.
    :param map_dag_j_to_dag_i: A dictionary that maps nodes from `dag_j` to nodes in `dag_i`.
    :param node: The node in `dag_j` to add to `dag_i`.
    :return: The updated `dag_i` with the added sub DAG.
    """
    if node in map_dag_j_to_dag_i:  # node was already processed
        return dag_i
    if dag_j.get_canonization(node) in dag_i.canonization_map:  # subDAG already in DAG
        map_dag_j_to_dag_i[node] = dag_i.canonization_map.get(dag_j.get_canonization(node))
        return dag_i
    # do the adding top down (since we have the canonization)
    if node in dag_j.get_leaves():  # if we are at leaf level, we can just add it
        new_node = dag_i.add_next_node(dag_j.h_node_attributes.get(node))
        map_dag_j_to_dag_i[node] = new_node
        dag_i.canonization[new_node] = dag_j.canonization[node]
        dag_i.node_heights[new_node] = dag_j.node_heights[node]
        dag_i.canonization_map[dag_i.canonization[new_node]] = new_node
        return dag_i

    # height = dag_j.get_height(node)  # we only need to check nodes with the same height
    # candidates = set(dag_i.roots)
    # while candidates:  # first try to add whole DAG
    #     new_candidates = set()
    #     for candidate in candidates:
    #         if dag_i.get_height(candidate) > height:  # height of current candidate is too big
    #             new_candidates.update(dag_i.get_children(candidate))
    #         elif dag_i.get_canonization(candidate) == dag_j.canonization.get(node):  # found the node
    #             map_dag_j_to_dag_i[node] = candidate
    #             return dag_i
    #     candidates = new_candidates
    #
    # # It was not possible to add whole DAG
    # add children first and then add new node + edges (including edge attributes and canonization!)
    for child in dag_j.get_children(node):  # add children first
        __add_sub_dag__(dag_i, dag_j, map_dag_j_to_dag_i, child)
    new_node = dag_i.add_next_node(dag_j.h_node_attributes.get(node))  # add new node
    for child in dag_j.get_children(node):  # add edges (including edge attributes and canonization!)
        real_child = map_dag_j_to_dag_i[child]
        if dag_i.is_edge_exists(new_node, real_child):
            # This should only happen when the DAG was not compressed
            dag_j_edge_attributes = set(dag_j.h_edge_attributes[(child, node)])
            dag_i_edge_attributes = set(dag_i.h_edge_attributes[(real_child, new_node)])
            intersection_between_dag_j_and_dag_i = dag_j_edge_attributes.intersection(dag_i_edge_attributes)
            difference_between_dag_j_and_dag_i = dag_j_edge_attributes.difference(dag_i_edge_attributes)
            if intersection_between_dag_j_and_dag_i:
                existing_multiplicities = dag_i.edge_multiplicities.get((real_child, new_node), {})
                for edge_attr, edge_multiplicity in existing_multiplicities.items():
                    if edge_attr in intersection_between_dag_j_and_dag_i:
                        dag_i.set_edge_multiplicity(
                            new_node,
                            real_child,
                            edge_attr,
                            edge_multiplicity + dag_j.edge_multiplicities[(child, node)][edge_attr],
                        )
            if difference_between_dag_j_and_dag_i:
                new_multiplicities = dag_j.edge_multiplicities.get((child, node), {})
                for edge_attr, edge_multiplicity in new_multiplicities.items():
                    if edge_attr in difference_between_dag_j_and_dag_i:
                        dag_i.add_edge(new_node, real_child, edge_attr, edge_multiplicity)
        else:
            new_multiplicities = dag_j.edge_multiplicities.get((child, node), {})
            for edge_attr, edge_multiplicity in new_multiplicities.items():
                dag_i.add_edge(new_node, real_child, edge_attr, edge_multiplicity)

    map_dag_j_to_dag_i[node] = new_node
    dag_i.canonization[new_node] = dag_j.canonization[node]
    dag_i.node_heights[new_node] = dag_j.node_heights[node]
    dag_i.canonization_map[dag_i.canonization[new_node]] = new_node
    return dag_i
