import unittest
from unittest import TestCase

from loguru import logger
from torch import cat, float, ones, randint, randperm, zeros, zeros_like
from torch.nn import Identity, Linear, Sequential
from torch.testing import assert_close
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from src.dag_gnn import ToCanonizedDirectedAcyclicGraph, ToCanonizedDirectedAcyclicGraphBatched
from task_graph.tudataset.dagmlp_model import DAGMLPLayer

N_JOBS = 1


class ModifiedGINConv(GINConv):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(ModifiedGINConv, self).__init__(nn, eps, train_eps, flow="target_to_source", **kwargs)

    def forward(self, x, features, edge_index, edge_weight=None, size=None):
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = out + (1 + self.eps) * features
        return self.nn(out)

    def message(self, x_j, edge_weight=None, reduce=sum):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class TestDAGMLP(TestCase):
    """
    Main test class for the DAG
    """

    def setUp(self):
        # Setup random graph arguments
        num_nodes = 10
        p_edges = 0.99
        num_features = 2
        num_classes = 2
        edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=True)
        edge_index, _ = remove_self_loops(edge_index)
        labels = randint(0, num_classes, (num_nodes,))
        features = ones(num_nodes, num_features)

        self.data = Data(
            x=features,
            edge_index=edge_index,
            y=labels,
            num_nodes=num_nodes,
            num_features=num_features,
            num_classes=num_classes,
        )

        self.num_layers = 5

    @logger.catch(reraise=True)
    def test_non_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with non-learnable parameters
        against a naive GINConv layer with non-learnable parameters.
        """
        gin_layer_without_learnable_parameters = ModifiedGINConv(Identity(), eps=0.0, train_eps=False)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(Identity(), eps=0.0, train_eps=False)

        x, edge_index = self.data.x, self.data.edge_index
        features = self.data.x.clone()

        gin_at_each_layer = [
            x,
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index)
            gin_at_each_layer += [
                x,
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraph(
            self.data.num_nodes, self.num_layers, self.num_layers, n_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform(self.data)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_multiplicities = dag_data.edge_multiplicities
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readout_at_each_layer = dag_data.dag_readout_at_each_layer
        dag_leaves_at_each_layer = dag_data.dag_leaves_at_each_layer

        leaves_0 = dag_leaves_at_each_layer[0]
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            edge_multiplicities_i = dag_edge_multiplicities[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, edge_multiplicities_i)

        dag_mlp_x_at_each_layer = []
        for i in range(self.num_layers + 1):
            readout_i = dag_readout_at_each_layer[i]
            dag_mlp_x_at_each_layer += [
                x[readout_i],
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i, dag_mlp_x_i)

    @logger.catch(reraise=True)
    def test_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with learnable parameters
        against a naive GINConv layer with learnable parameters.
        """
        # Initialize the MLP that will be used in the GINConv and DAGMLPLayer
        mlp = Sequential(Linear(self.data.num_features, self.data.num_features, bias=False))
        gin_layer_without_learnable_parameters = ModifiedGINConv(mlp)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(mlp)

        x, edge_index = self.data.x, self.data.edge_index
        features = self.data.x.clone()

        gin_at_each_layer = [
            x,
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index)
            gin_at_each_layer += [
                x,
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraph(
            self.data.num_nodes, self.num_layers, self.num_layers, n_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform(self.data)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_multiplicities = dag_data.edge_multiplicities
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readout_at_each_layer = dag_data.dag_readout_at_each_layer
        dag_leaves_at_each_layer = dag_data.dag_leaves_at_each_layer

        leaves_0 = dag_leaves_at_each_layer[0]
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            edge_multiplicities_i = dag_edge_multiplicities[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, edge_multiplicities_i)

        dag_mlp_x_at_each_layer = []
        for i in range(self.num_layers + 1):
            readout_i = dag_readout_at_each_layer[i]
            dag_mlp_x_at_each_layer += [
                x[readout_i],
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i.trunc(), dag_mlp_x_i.trunc())


class TestBatchDAGMLP(TestCase):
    """
    Main test class for the DAG
    """

    def setUp(self):
        # Setup random graph arguments
        num_graphs = 10
        num_nodes = 10
        p_edges = 0.99
        num_features = 2
        num_classes = 2

        self.data_list = []
        for graph_i in range(num_graphs):
            edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=True)
            edge_index, _ = remove_self_loops(edge_index)
            label = randint(0, num_classes, (num_nodes,))
            features = ones(num_nodes, num_features)

            data = Data(
                x=features,
                edge_index=edge_index,
                y=label,
                num_nodes=num_nodes,
                num_features=num_features,
                num_classes=num_classes,
            )
            self.data_list += [
                data,
            ]

        self.num_layers = 5
        self.pooling = global_add_pool

    @logger.catch(reraise=True)
    def test_non_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with non-learnable parameters
        against a naive GINConv layer with non-learnable parameters.
        """
        gin_layer_without_learnable_parameters = ModifiedGINConv(Identity(), eps=0.0, train_eps=False)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(Identity(), eps=0.0, train_eps=False)

        batched_data = Batch.from_data_list(self.data_list)
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        features = batched_data.x.clone()

        gin_at_each_layer = [
            self.pooling(x, batch),
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index)
            gin_at_each_layer += [
                self.pooling(x, batch),
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraphBatched(
            num_layers=self.num_layers, k=self.num_layers, n_graphs_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform.from_data_list([*range(len(self.data_list))], self.data_list)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_multiplicities = dag_data.edge_multiplicities
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readouts_per_graph = [*map(list, zip(*dag_data.dag_readout_at_each_layer))]  # Transpose the list of lists
        dag_leaves_at_each_layer_per_graph = dag_data.dag_leaves_at_each_layer

        leaves_0 = cat([leaves_i[0] for leaves_i in dag_leaves_at_each_layer_per_graph])
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            edge_multiplicities_i = dag_edge_multiplicities[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, edge_multiplicities_i)

        dag_mlp_x_at_each_layer = []

        for layer_i in range(self.num_layers + 1):
            pooled_graphs = [
                *map(lambda readouts: self.pooling(x[readouts], zeros_like(readouts)), dag_readouts_per_graph[layer_i])
            ]
            dag_mlp_x_at_each_layer += [
                cat(pooled_graphs, dim=0),
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i, dag_mlp_x_i)

    @logger.catch(reraise=True)
    def test_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with learnable parameters
        against a naive GINConv layer with learnable parameters.
        """
        # Initialize the MLP that will be used in the GINConv and DAGMLPLayer
        mlp = Sequential(Linear(self.data_list[0].num_features, self.data_list[0].num_features, bias=False))
        gin_layer_without_learnable_parameters = ModifiedGINConv(mlp)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(mlp)

        batched_data = Batch.from_data_list(self.data_list)
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        features = batched_data.x.clone()

        gin_at_each_layer = [
            self.pooling(x, batch),
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index)
            gin_at_each_layer += [
                self.pooling(x, batch),
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraphBatched(
            num_layers=self.num_layers, k=self.num_layers, n_graphs_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform.from_data_list([*range(len(self.data_list))], self.data_list)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_multiplicities = dag_data.edge_multiplicities
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readouts_per_graph = [*map(list, zip(*dag_data.dag_readout_at_each_layer))]  # Transpose the list of lists
        dag_leaves_at_each_layer_per_graph = dag_data.dag_leaves_at_each_layer

        leaves_0 = cat([leaves_i[0] for leaves_i in dag_leaves_at_each_layer_per_graph])
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            edge_multiplicities_i = dag_edge_multiplicities[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, edge_multiplicities_i)

        dag_mlp_x_at_each_layer = []
        for layer_i in range(self.num_layers + 1):
            pooled_graphs = [
                *map(lambda readouts: self.pooling(x[readouts], zeros_like(readouts)), dag_readouts_per_graph[layer_i])
            ]
            dag_mlp_x_at_each_layer += [
                cat(pooled_graphs, dim=0),
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i.trunc(), dag_mlp_x_i.trunc())


def generate_symmetric_edge_attributes(edge_index, max_value=3):
    # Create a dictionary to hold unique edges and their attributes
    edge_dict = {}
    # Create a tensor to hold the attributes for all edges
    edge_attr = zeros(edge_index.shape[1], dtype=float)
    # Create a pool of unique random integers
    unique_rand_ints = randperm(max_value)[: edge_index.shape[1]] + 1
    for i in range(edge_index.shape[1]):
        node1, node2 = edge_index[:, i]
        # Use a tuple of the sorted nodes as the key, so (i, j) and (j, i) result in the same key
        edge_key_i = (node1.item(), node2.item())
        edge_key_j = (node2.item(), node1.item())
        # If this edge (or its reverse) has not been processed yet, draw an integer from the pool
        if edge_key_i not in edge_dict and edge_key_j not in edge_dict:
            edge_dict[edge_key_i] = unique_rand_ints[i % max_value].item()
            edge_dict[edge_key_j] = unique_rand_ints[i % max_value].item()
    edge_index_as_list = edge_index.t().tolist()
    for e, v in edge_dict.items():
        if [*e] in edge_index_as_list:
            index = edge_index_as_list.index([*e])
            edge_attr[index] = v
    return edge_attr


class TestEdgeDAGMLP(TestCase):
    """
    Main test class for the DAG
    """

    def setUp(self):
        # Setup random graph arguments
        num_nodes = 10
        p_edges = 0.8
        num_classes = 2
        node_num_features = 2
        edge_num_features = 3

        edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=True)
        edge_index, _ = remove_self_loops(edge_index)
        edge_attr = generate_symmetric_edge_attributes(edge_index)  # .repeat(edge_num_features, 1).T
        labels = randint(0, num_classes, (num_nodes,))
        features = ones(num_nodes, node_num_features)

        self.data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            num_nodes=num_nodes,
            num_features=node_num_features,
            num_classes=num_classes,
        )

        self.num_layers = 2

    @logger.catch(reraise=True)
    def test_non_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with learnable parameters
        against a naive GINConv layer with learnable parameters.
        """
        gin_layer_without_learnable_parameters = ModifiedGINConv(Identity(), eps=0.0, train_eps=False)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(Identity(), eps=0.0, train_eps=False)

        x, edge_index, edge_attr = self.data.x, self.data.edge_index, self.data.edge_attr
        features = self.data.x.clone()

        gin_at_each_layer = [
            x,
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index, edge_attr)
            gin_at_each_layer += [
                x,
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraph(
            self.data.num_nodes, self.num_layers, self.num_layers, n_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform(self.data)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_attr = dag_data.dag_edge_attr
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readout_at_each_layer = dag_data.dag_readout_at_each_layer
        dag_leaves_at_each_layer = dag_data.dag_leaves_at_each_layer

        leaves_0 = dag_leaves_at_each_layer[0]
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            dag_edge_attr_i = dag_edge_attr[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, dag_edge_attr_i)

        dag_mlp_x_at_each_layer = []
        for i in range(self.num_layers + 1):
            readout_i = dag_readout_at_each_layer[i]
            dag_mlp_x_at_each_layer += [
                x[readout_i],
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i, dag_mlp_x_i)

    @logger.catch(reraise=True)
    def test_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with learnable parameters
        against a naive GINConv layer with learnable parameters.
        """
        # Initialize the MLP that will be used in the GINConv and DAGMLPLayer
        mlp = Sequential(Linear(self.data.num_features, self.data.num_features, bias=False))
        gin_layer_without_learnable_parameters = ModifiedGINConv(mlp, eps=0.0, train_eps=False)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(mlp, eps=0.0, train_eps=False)

        x, edge_index, edge_attr = self.data.x, self.data.edge_index, self.data.edge_attr
        features = self.data.x.clone()

        gin_at_each_layer = [
            x,
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index, edge_attr)
            gin_at_each_layer += [
                x,
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraph(
            self.data.num_nodes, self.num_layers, self.num_layers, n_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform(self.data)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_attr = dag_data.dag_edge_attr
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readout_at_each_layer = dag_data.dag_readout_at_each_layer
        dag_leaves_at_each_layer = dag_data.dag_leaves_at_each_layer

        leaves_0 = dag_leaves_at_each_layer[0]
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            dag_edge_attr_i = dag_edge_attr[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, dag_edge_attr_i)

        dag_mlp_x_at_each_layer = []
        for i in range(self.num_layers + 1):
            readout_i = dag_readout_at_each_layer[i]
            dag_mlp_x_at_each_layer += [
                x[readout_i],
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i.trunc(), dag_mlp_x_i.trunc())


class TestBatchEdgeDAGMLP(TestCase):
    """
    Main test class for the DAG
    """

    def setUp(self):
        # Setup random graph arguments
        num_graphs = 10
        num_nodes = 10
        p_edges = 0.99
        node_num_features = 2
        edge_num_features = 3
        num_classes = 2

        self.data_list = []
        for graph_i in range(num_graphs):
            edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=True)
            edge_index, _ = remove_self_loops(edge_index)
            edge_attr = generate_symmetric_edge_attributes(edge_index)  # .repeat(edge_num_features, 1).T
            label = randint(0, num_classes, (num_nodes,))
            features = ones(num_nodes, node_num_features)

            data = Data(
                x=features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=label,
                num_nodes=num_nodes,
                num_features=node_num_features,
                num_classes=num_classes,
            )
            self.data_list += [
                data,
            ]

        self.num_layers = 2
        self.pooling = global_add_pool

    @logger.catch(reraise=True)
    def test_non_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with non-learnable parameters
        against a naive GINConv layer with non-learnable parameters.
        """
        gin_layer_without_learnable_parameters = ModifiedGINConv(Identity(), eps=0.0, train_eps=False)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(Identity(), eps=0.0, train_eps=False)

        batched_data = Batch.from_data_list(self.data_list)
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )
        features = batched_data.x.clone()

        gin_at_each_layer = [
            self.pooling(x, batch),
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index, edge_attr)
            gin_at_each_layer += [
                self.pooling(x, batch),
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraphBatched(
            num_layers=self.num_layers, k=self.num_layers, n_graphs_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform.from_data_list([*range(len(self.data_list))], self.data_list)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_attr = dag_data.dag_edge_attr
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readouts_per_graph = [*map(list, zip(*dag_data.dag_readout_at_each_layer))]  # Transpose the list of lists
        dag_leaves_at_each_layer_per_graph = dag_data.dag_leaves_at_each_layer

        leaves_0 = cat([leaves_i[0] for leaves_i in dag_leaves_at_each_layer_per_graph])
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            dag_edge_attr_i = dag_edge_attr[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, dag_edge_attr_i)

        dag_mlp_x_at_each_layer = []

        for layer_i in range(self.num_layers + 1):
            pooled_graphs = [
                *map(lambda readouts: self.pooling(x[readouts], zeros_like(readouts)), dag_readouts_per_graph[layer_i])
            ]
            dag_mlp_x_at_each_layer += [
                cat(pooled_graphs, dim=0),
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i, dag_mlp_x_i)

    @logger.catch(reraise=True)
    def test_learnable_canonized_dag_mlp(self):
        """
        This test validates the functionality of a DAG MLP layer with learnable parameters
        against a naive GINConv layer with learnable parameters.
        """
        # Initialize the MLP that will be used in the GINConv and DAGMLPLayer
        mlp = Sequential(Linear(self.data_list[0].num_features, self.data_list[0].num_features, bias=False))
        gin_layer_without_learnable_parameters = ModifiedGINConv(mlp)
        dag_mlp_layer_without_learnable_parameters = DAGMLPLayer(mlp)

        batched_data = Batch.from_data_list(self.data_list)
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )
        features = batched_data.x.clone()

        gin_at_each_layer = [
            self.pooling(x, batch),
        ]
        for i in range(self.num_layers):
            x = gin_layer_without_learnable_parameters(x, features, edge_index, edge_attr)
            gin_at_each_layer += [
                self.pooling(x, batch),
            ]

        dag_transform = ToCanonizedDirectedAcyclicGraphBatched(
            num_layers=self.num_layers, k=self.num_layers, n_graphs_jobs=N_JOBS, verbose=False
        )
        dag_data = dag_transform.from_data_list([*range(len(self.data_list))], self.data_list)

        # Extract the DAG data
        dag_x = dag_data.dag_x
        dag_edge_index = dag_data.dag_edge_index
        dag_edge_attr = dag_data.dag_edge_attr
        dag_layers_mask = dag_data.dag_layers_mask
        dag_readouts_per_graph = [*map(list, zip(*dag_data.dag_readout_at_each_layer))]  # Transpose the list of lists
        dag_leaves_at_each_layer_per_graph = dag_data.dag_leaves_at_each_layer

        leaves_0 = cat([leaves_i[0] for leaves_i in dag_leaves_at_each_layer_per_graph])
        x = zeros_like(dag_x)
        x[leaves_0] = dag_x[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            dag_edge_attr_i = dag_edge_attr[dag_layers_mask == i]
            x = dag_mlp_layer_without_learnable_parameters(x, dag_x, edge_index_i, dag_edge_attr_i)

        dag_mlp_x_at_each_layer = []
        for layer_i in range(self.num_layers + 1):
            pooled_graphs = [
                *map(lambda readouts: self.pooling(x[readouts], zeros_like(readouts)), dag_readouts_per_graph[layer_i])
            ]
            dag_mlp_x_at_each_layer += [
                cat(pooled_graphs, dim=0),
            ]

        # Compare the results
        for gin_x_i, dag_mlp_x_i in zip(gin_at_each_layer, dag_mlp_x_at_each_layer):
            assert_close(gin_x_i.trunc(), dag_mlp_x_i.trunc())


if __name__ == "__main__":
    unittest.main()
