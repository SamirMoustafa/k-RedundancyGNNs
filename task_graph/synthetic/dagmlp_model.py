from torch import Tensor, cat, ones, zeros_like
from torch.nn import Linear, Module, ModuleDict, Parameter, ReLU, Sequential
from torch.nn.functional import relu
from torch_geometric.nn import MessagePassing, global_add_pool


def target_nodes_self_loops(edge_index):
    """
    Generates self-loops for target nodes

    :param edge_index: Edge index tensor
    :return: Tensor of shape (2, edge_index.shape[1]) with self-loops for target nodes
    """
    loop_index = edge_index[1].unique().repeat(2, 1)
    return loop_index


class DAGMLPLayer(MessagePassing):
    def __init__(self, nn, eps=0.0, train_eps=False):
        super().__init__()
        self.nn = nn
        self.eps = Parameter(Tensor([eps]), requires_grad=train_eps)

    def reset_parameters(self):
        for layer in self.nn:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.eps.data.fill_(0.0)

    def forward(self, x: Tensor, features: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        target_nodes_id = edge_index[1].unique()
        # Compute (1 + ε_i) * L_i * F', which corresponds to the term (1 + ε_i) * L_i * F' in the equation
        L = target_nodes_self_loops(edge_index)  # L_i matrix
        e_L_F = (1 + self.eps) * self.propagate(L, x=features, edge_weight=ones(L.shape[1], device=L.device))
        # Compute E_i * X^[i-1], corresponds to E_i * X^[i-1] in the equation
        E_X = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        # Combine (1 + ε_i) * L_i * F' + E_i * X^[i-1]
        e_L_F_plus_E_X = e_L_F + E_X
        # Apply MLP to combined term and update only target nodes
        # This is equivalent to applying MLP_i in the equation to the term inside the parentheses
        e_L_F_plus_E_X[target_nodes_id] = self.nn(e_L_F_plus_E_X[target_nodes_id])
        # Add X^[i-1] (This corresponds to the last term X^[i-1] in the equation)
        e_L_F_plus_E_X += x
        # The output is now X^[i] according to the equation
        return e_L_F_plus_E_X

    def message(self, x_j: Tensor, edge_weight: Tensor = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return "{}(nn={}, eps={})".format(self.__class__.__name__, self.nn, round(self.eps.item(), 3))


class DAGMLP(Module):
    def __init__(self, dim_features, dim_embedding, dim_target, num_layers, predict=True):
        super(DAGMLP, self).__init__()
        self.num_layers = num_layers
        self.dim_embedding = dim_embedding
        self.pooling = global_add_pool
        self.predict = predict

        self.layers_dict = self.build_dag_layers(self.dim_embedding, False)

        self.node_features_transformation = Linear(dim_features, self.dim_embedding)

        # For concatenated node features
        self.fc1 = Linear((self.num_layers + 1) * self.dim_embedding, self.dim_embedding)
        self.fc2 = Linear(self.dim_embedding, dim_target)

    def build_dag_layers(self, dim_embedding, train_eps):
        layers = ModuleDict()
        for i in range(self.num_layers):
            layers[f"nn_{i}"] = Sequential(
                Linear(dim_embedding, dim_embedding), ReLU(), Linear(dim_embedding, dim_embedding), ReLU()
            )
            layers[f"dag_mlp_{i}"] = DAGMLPLayer(layers[f"nn_{i}"], train_eps=train_eps)
        return layers

    def reset_parameters(self):
        for layer in self.layers_dict.values():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.node_features_transformation.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        dag_edge_attr = data.dag_edge_attr
        dag_edge_index, dag_layers_mask = data.dag_edge_index, data.dag_layers_mask
        dag_readouts, dag_leaves_at_each_layer = data.dag_readout_at_each_layer, data.dag_leaves_at_each_layer
        dag_x, batch, leaves_0 = data.dag_x, data.batch, dag_leaves_at_each_layer[0]

        feature = dag_x.clone()
        feature = relu(self.node_features_transformation(feature))

        x = zeros_like(feature)
        x[leaves_0] = feature[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            dag_edge_attr_i = dag_edge_attr[dag_layers_mask == i]
            x = self.layers_dict[f"dag_mlp_{i}"](x, feature, edge_index_i, dag_edge_attr_i)

        x_at_each_layer = []
        for i in range(self.num_layers + 1):
            readout_i = dag_readouts[i]
            x_at_each_layer += [
                self.pooling(x[readout_i], batch[readout_i]),
            ]

        x = cat(x_at_each_layer, dim=1)
        if self.predict:
            x = relu(self.fc1(x))
            x = self.fc2(x)
        return x
