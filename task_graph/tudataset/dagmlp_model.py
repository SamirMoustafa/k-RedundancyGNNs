from torch import Tensor, ones, stack, zeros_like
from torch.nn import BatchNorm1d, Dropout, Linear, Module, ModuleDict, ModuleList, Parameter, ReLU, Sequential
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool


class PypassBatchNorm1d(Module):
    # It is the same as BatchNorm1d, but it does not apply normalization to batches of size 1
    # This was implemented to cover a runtime error cause by the dataset NCI1 in TUDataset.
    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None
    ):
        super(PypassBatchNorm1d, self).__init__()
        self.batch_norm = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats).to(
            device=device, dtype=dtype
        )

    def forward(self, x):
        if x.size(0) == 1:
            return x
        else:
            return self.batch_norm(x)


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
    def __init__(self, dim_features, dim_target, **config):
        super(DAGMLP, self).__init__()
        self.num_layers = config["num_layers"]
        self.dim_embedding = config["dim_embedding"]
        self.combine_multi_height = config["combine_multi_height"]
        self.use_linear_after_readout = config["use_linear_after_readout"]
        dropout = config["dropout"]
        train_eps = config["train_eps"]

        if config["aggregation"] == "sum":
            self.pooling = global_add_pool
        elif config["aggregation"] == "mean":
            self.pooling = global_mean_pool
        elif config["aggregation"] == "max":
            self.pooling = global_max_pool
        else:
            raise NotImplementedError("Aggregation method not implemented")
        if self.use_linear_after_readout and not self.combine_multi_height:
            raise ValueError("use_linear_after_readout can only be used with combine_multi_height")

        self.node_features_transformation = Sequential(
            Linear(dim_features, self.dim_embedding),
            PypassBatchNorm1d(self.dim_embedding),
            ReLU(),
            Linear(self.dim_embedding, self.dim_embedding),
            PypassBatchNorm1d(self.dim_embedding),
            ReLU(),
        )
        self.dag_layers = self.build_layers(self.dim_embedding, train_eps)
        self.dropout = Dropout(dropout)

        if self.use_linear_after_readout:
            self.readout_linear = ModuleList(
                [Linear(self.dim_embedding, self.dim_embedding) for _ in range(self.num_layers + 1)]
            )

        self.last_layer = Linear(self.dim_embedding, dim_target)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.dag_layers.values():
            layer.reset_parameters()
        xavier_uniform_(self.last_layer.weight)
        zeros_(self.last_layer.bias)

    def build_layers(self, dim_embedding, train_eps):
        dag_layers = ModuleDict()
        for i in range(self.num_layers):
            mlp_i = Sequential(
                Linear(dim_embedding, dim_embedding),
                PypassBatchNorm1d(dim_embedding),
                ReLU(),
                Linear(dim_embedding, dim_embedding),
                PypassBatchNorm1d(dim_embedding),
                ReLU(),
            )
            dag_layers[f"dag_mlp_{i}"] = DAGMLPLayer(mlp_i, train_eps=train_eps)
        return dag_layers

    def forward(self, data):
        dag_edge_attr = data.edge_multiplicities
        dag_edge_index, dag_layers_mask = data.dag_edge_index, data.dag_layers_mask
        dag_readouts, dag_leaves_at_each_layer = data.dag_readout_at_each_layer, data.dag_leaves_at_each_layer
        dag_x, batch, leaves_0 = data.dag_x, data.batch, dag_leaves_at_each_layer[0]

        feature = self.node_features_transformation(dag_x)

        x = zeros_like(feature)
        x[leaves_0] = feature[leaves_0]

        for i in range(self.num_layers):
            edge_index_i = dag_edge_index[:, dag_layers_mask == i]
            dag_edge_attr_i = dag_edge_attr[dag_layers_mask == i]
            x = self.dropout(self.dag_layers[f"dag_mlp_{i}"](x, feature, edge_index_i, dag_edge_attr_i))

        if self.combine_multi_height:
            layer_outputs = [self.pooling(x[readout_i], batch[readout_i]) for i, readout_i in enumerate(dag_readouts)]
            if self.use_linear_after_readout:
                layer_outputs = [
                    self.dropout(self.readout_linear[i](layer_out_i)) for i, layer_out_i in enumerate(layer_outputs)
                ]
            x = stack(layer_outputs, dim=1).mean(dim=1)
        else:
            x = self.pooling(x[dag_readouts[-1]], batch[dag_readouts[-1]])
        x = self.last_layer(x)
        return x
