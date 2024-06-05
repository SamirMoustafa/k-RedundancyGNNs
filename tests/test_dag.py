import unittest
from unittest import TestCase

from loguru import logger
from torch import float32, ones, randint, sparse, sparse_coo_tensor, stack, zeros
from torch.testing import assert_close
from torch_geometric.data import Data
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, sort_edge_index

from src.dag_gnn import ToNoneCanonizedDirectedAcyclicGraph

N_JOBS = 1


def sparse_dense_mm(edge_index, x, edge_weight=None):
    v = ones(edge_index.shape[1], requires_grad=True).to(x.device) if edge_weight is None else edge_weight
    sparse_matrix = sparse_coo_tensor(edge_index, v, size=(x.size(0), x.size(0)), requires_grad=True)
    return sparse.mm(sparse_matrix, x)


def sparse_t_dense_mm(edge_index, x, edge_weight=None):
    edge_index_t = stack([edge_index[1], edge_index[0]], dim=0)
    y = sparse_dense_mm(edge_index_t, x, edge_weight)
    return y


def generate_erdos_renyi_graph(num_nodes, p_edges, num_features, num_classes):
    """
    Generate a random graph with an Erdos-Renyi model

    :param num_nodes: Number of nodes
    :param p_edges: Probability of an edge between two nodes
    :param num_features: Length of the feature vector
    :param num_classes: Number of classes
    :return: Tuple of edge_index, features, labels
    """
    edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=False)
    edge_index, _ = remove_self_loops(edge_index)
    features = randint(0, 10, (num_nodes, num_features), dtype=float32)
    labels = randint(0, num_classes, (num_nodes,))
    return edge_index, features, labels


class TestDAG(TestCase):
    """
    Main test class for the DAG
    """

    def setUp(self):
        # Setup random graph arguments
        num_nodes = 10
        p_edges = 0.1
        num_features = 10
        num_classes = 2
        # Generate a random graph with Erdos-Renyi model
        edge_index, features, labels = generate_erdos_renyi_graph(num_nodes, p_edges, num_features, num_classes)
        self.data = Data(
            x=features,
            edge_index=edge_index,
            y=labels,
            num_nodes=num_nodes,
            num_features=num_features,
            num_classes=num_classes,
        )
        # Define the number of layers
        self.num_layers = 10

    @logger.catch(reraise=True)
    def test_directed_acyclic_graph_creation(self):
        """
        Test that the DAG is created correctly for different number of layers, and it's all possible values of k
        """
        # Sort the original-edge index
        sorted_edge_index = sort_edge_index(self.data.edge_index.clone())

        # Iterate over the number of layers
        for num_layer in range(self.num_layers):
            # Iterate over all possible values of k
            for k in range(num_layer + 1):
                # Create the DAG transform and apply it to the data
                dag_transform = ToNoneCanonizedDirectedAcyclicGraph(
                    self.data.num_nodes, num_layers=num_layer + 1, k=k, n_jobs=N_JOBS, verbose=False
                )
                dag_data = dag_transform(self.data)
                # Extract the DAG-edge index, and map the DAG nodes id to the original nodes order
                dag_edge_index = dag_data.dag_edge_index
                dag_nodes_order = dag_data.dag_nodes_order
                edge_index_src = dag_nodes_order[dag_edge_index[1]]
                edge_index_dsc = dag_nodes_order[dag_edge_index[0]]
                edge_index_from_dag = stack([edge_index_src, edge_index_dsc], dim=0)
                # Sort and remove repeated edges
                unique_sorted_edge_index_from_dag = sort_edge_index(edge_index_from_dag).unique(dim=1)

                # Check that the original-edge index is the same as the edge index from the DAG
                assert_close(sorted_edge_index, unique_sorted_edge_index_from_dag)

    @logger.catch(reraise=True)
    def test_naive_message_passing(self):
        """
        Test that the naive message passing is the same for SimpleConv and sparse_t_dense_mm
        """
        message_passing_operator = SimpleConv(aggr="add")

        x, edge_index = self.data.x, self.data.edge_index
        for num_layer_i in range(self.num_layers):
            x = message_passing_operator(x, edge_index)
        x_after_i_hops = x.clone()

        x, edge_index = self.data.x, self.data.edge_index
        for num_layer_i in range(self.num_layers):
            x = sparse_t_dense_mm(edge_index, x)
        x_after_i_hops_sparse_t_dense_mm = x.clone()

        assert_close(x_after_i_hops, x_after_i_hops_sparse_t_dense_mm)

    @logger.catch(reraise=True)
    def test_dag_message_passing(self):
        """
        Test that the message passing is the same for the original graph and the DAG
        """
        message_passing_operator = SimpleConv(aggr="add")
        x, edge_index = self.data.x, self.data.edge_index

        # Apply the message passing operator for i-hops
        x_n_hops = x.clone()
        for num_layer_i in range(self.num_layers):
            x_n_hops = message_passing_operator(x_n_hops, edge_index)

        # Create the DAG transform and apply it to the data
        dag_transform = ToNoneCanonizedDirectedAcyclicGraph(
            self.data.num_nodes, num_layers=self.num_layers, k=self.num_layers, n_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform(self.data)

        # Extract the DAG data
        dag_roots = dag_data.dag_roots
        dag_leaves = dag_data.dag_leaves
        dag_edge_index = dag_data.dag_edge_index
        dag_nodes_order = dag_data.dag_nodes_order
        dag_layers_mask = dag_data.dag_layers_mask

        # Check that the number of layers is the same as the number of masked indices
        self.assertEqual(self.num_layers, dag_layers_mask.max().item() + 1)

        # Map the DAG nodes id to the original nodes features
        x_n_dag_levels = zeros(len(dag_nodes_order), self.data.num_features)
        x_n_dag_levels[dag_leaves] = x.clone()

        # Iterate over the masked indices (layers)
        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            x_n_dag_levels = sparse_t_dense_mm(edge_index_i, x_n_dag_levels)

        # Extract the DAG updated leaves
        x_n_dag_levels = x_n_dag_levels[dag_roots]

        # Check that the original message passing is the same as the message passing from the DAG
        assert_close(x_n_hops, x_n_dag_levels)


if __name__ == "__main__":
    unittest.main()
