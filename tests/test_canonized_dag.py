import unittest
from unittest import TestCase

from loguru import logger
from test_dag import sparse_t_dense_mm
from torch import ones, randint, zeros_like
from torch.testing import assert_close
from torch_geometric.data import Data
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from src.dag_gnn import ToCanonizedDirectedAcyclicGraph

N_JOBS = 1


class TestDAG(TestCase):
    """
    Main test class for the DAG
    """

    def setUp(self):
        # Setup random graph arguments
        num_nodes = 10
        p_edges = 0.7
        num_features = 2
        num_classes = 2
        edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=True)
        edge_index, _ = remove_self_loops(edge_index)
        edge_attr = edge_index.sum(dim=0).float()

        labels = randint(0, num_classes, (num_nodes,))
        features = ones(num_nodes, num_features)

        self.data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            num_nodes=num_nodes,
            num_features=num_features,
            num_classes=num_classes,
        )
        # Define the number of layers
        self.num_layers = 3

    @logger.catch(reraise=True)
    def test_naive_message_passing(self):
        """
        Test that the naive message passing is the same for SimpleConv and sparse_t_dense_mm
        """
        message_passing_operator = SimpleConv(aggr="add")

        x, edge_index, edge_attr = self.data.x, self.data.edge_index, self.data.edge_attr
        for num_layer_i in range(self.num_layers):
            x = message_passing_operator(x, edge_index, edge_attr)
        x_after_i_hops = x.clone()

        x, edge_index = self.data.x, self.data.edge_index
        for num_layer_i in range(self.num_layers):
            x = sparse_t_dense_mm(edge_index, x, edge_attr)
        x_after_i_hops_sparse_t_dense_mm = x.clone()

        assert_close(x_after_i_hops, x_after_i_hops_sparse_t_dense_mm)

    @logger.catch(reraise=True)
    def test_dag_message_passing(self):
        """
        Test that the message passing is the same for the original graph and the DAG
        """
        message_passing_operator = SimpleConv(aggr="add", flow="target_to_source")
        x, edge_index, edge_attr = self.data.x, self.data.edge_index, self.data.edge_attr

        # Apply the message passing operator for i-hops
        x_at_each_layer = [
            x,
        ]
        for num_layer_i in range(self.num_layers):
            x = message_passing_operator(x, edge_index, edge_attr)
            x_at_each_layer += [
                x,
            ]

        # Create the DAG transform and apply it to the data
        dag_transform = ToCanonizedDirectedAcyclicGraph(
            self.data.num_nodes, num_layers=self.num_layers, k=self.num_layers, n_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform(self.data)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_multiplicities = dag_data.edge_multiplicities
        dag_edge_attr = dag_data.dag_edge_attr
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readout_at_each_layer = dag_data.dag_readout_at_each_layer
        dag_leaves_at_each_layer = dag_data.dag_leaves_at_each_layer

        leaves_0 = dag_leaves_at_each_layer[0]
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]
        edge_weights = dag_edge_attr * dag_edge_multiplicities

        # Check that the number of layers is the same as the number of masked indices
        self.assertEqual(self.num_layers, dag_layers_mask.max().item() + 1)
        self.assertEqual(self.num_layers + 1, len(dag_readout_at_each_layer))

        dag_x_at_each_layer = []
        dag_x_at_each_layer += [
            x[dag_readout_at_each_layer[0]],
        ]

        # Iterate over the masked indices (layers)
        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            edge_weight_i = edge_weights[dag_layers_mask == i]
            x = sparse_t_dense_mm(edge_index_i, x, edge_weight_i)
            dag_x_at_each_layer += [
                x[dag_readout_at_each_layer[i + 1]],
            ]

        # Compare the results
        for x_i, dag_x_i in zip(x_at_each_layer, dag_x_at_each_layer):
            assert_close(x_i, dag_x_i)


if __name__ == "__main__":
    unittest.main()
