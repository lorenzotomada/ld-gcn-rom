import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class Encoder(torch.nn.Module):
    """
    Graph Encoder Module for latent space projection.
    This class compresses graph-structured data into a low-dimensional bottleneck. 
    It supports multiple graph convolution operators and utilizes a Feed-Forward 
    Network (FFN) to finalize the encoding after global node-feature flattening.

    Mathematical Formulation:
    1. Graph Convolution: $\mathbf{X}_{l+1} = \sigma(\text{GNN}(\mathbf{X}_l, \mathbf{A}, \mathbf{E}))$
    2. Flattening: $\mathbf{z}_{flat} = \text{vec}(\mathbf{X}_{depth})$
    3. Latent Projection: $\mathbf{z}_{lat} = \mathbf{W}_2(\sigma(\mathbf{W}_1 \mathbf{z}_{flat} + \mathbf{b}_1)) + \mathbf{b}_2$

    Args:
        hidden_channels (list[int]): Feature dimensions for each GNN layer. 
            Example: [3, 16, 32] where 3 is the input node feature size.
        bottleneck (int): Dimensionality of the final latent vector.
        input_size (int): Number of nodes per graph (assumes fixed mesh/topology).
        ffn (int): Hidden dimension of the projection MLP.
        skip (bool): If True, performs element-wise addition of input features to 
            each hidden layer's output (Residual Connection).
        act (function): Non-linear activation function. Defaults to F.elu.
        conv (str): Type of GNN operator. Options: 'GMMConv', 'ChebConv', 'GCNConv', 'GATConv'.
    """

    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip, act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.down_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            if self.conv=='GMMConv':
                self.down_convs.append(gnn.GMMConv(self.hidden_channels[i], self.hidden_channels[i+1], dim=1, kernel_size=5))
            elif self.conv=='ChebConv':
                self.down_convs.append(gnn.ChebConv(self.hidden_channels[i], self.hidden_channels[i+1], K=5))
            elif self.conv=='GCNConv':
                self.down_convs.append(gnn.GCNConv(self.hidden_channels[i], self.hidden_channels[i+1]))
            elif self.conv=='GATConv':
                self.down_convs.append(gnn.GATConv(self.hidden_channels[i], self.hidden_channels[i+1]))
            else:
                raise NotImplementedError('Invalid convolution selected. Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')

        self.fc_in1 = nn.Linear(self.input_size*self.hidden_channels[-1], self.ffn)
        self.fc_in2 = nn.Linear(self.ffn, self.bottleneck)
        self.reset_parameters()

    def encoder(self, data):
        """
        Executes the encoding pass.

        Args:
            data (torch_geometric.data.Data): Batch object containing:
                - x: Node feature matrix [Nodes*Batch, Features]
                - edge_index: Graph connectivity [2, Edges]
                - edge_weight / edge_attr: Edge features for convolution.

        Returns:
            torch.Tensor: Latent representations [Batch, Bottleneck].
        """
        x = data.x
        idx = 0
        for layer in self.down_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = self.act(layer(x, data.edge_index, data.edge_weight))
            elif self.conv in ['GATConv']:
                x = self.act(layer(x, data.edge_index, data.edge_attr))
            if self.skip:
                x = x + data.x
            idx += 1

        x = x.reshape(data.num_graphs, self.input_size * self.hidden_channels[-1])
        x = self.act(self.fc_in1(x))
        x = self.fc_in2(x)
        return x

    def reset_parameters(self):
        """Initializes weights using Kaiming Uniform and sets biases to zero."""
        for conv in self.down_convs:
            conv.reset_parameters()
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_uniform_(param)

    def forward(self,data):
        x = self.encoder(data)
        return x


class Decoder(torch.nn.Module):
    """
    Graph Decoder Module for field reconstruction.

    This class mirrors the Encoder, mapping the latent bottleneck back to the 
    original graph node feature space. It first expands the bottleneck via 
    MLP layers and then applies "up-convolutions" (reversed hidden channel sizes).

    Args:
        hidden_channels (list[int]): Feature dimensions for reconstruction layers. 
            Mirror of Encoder channels.
        bottleneck (int): Dimensionality of the input latent vector.
        input_size (int): Number of nodes per graph.
        ffn (int): Hidden dimension of the expansion MLP.
        skip (bool): If True, adds the expanded bottleneck features back to 
            each GNN layer's output.
        act (function): Non-linear activation function.
        conv (str): Type of GNN operator. Matches Encoder selection.
    """

    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip, act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.fc_out1 = nn.Linear(self.bottleneck, self.ffn)
        self.fc_out2 = nn.Linear(self.ffn, self.input_size * self.hidden_channels[-1])

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            if self.conv=='GMMConv':
                self.up_convs.append(gnn.GMMConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2], dim=1, kernel_size=5))
            elif self.conv=='ChebConv':
                self.up_convs.append(gnn.ChebConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2], K=5))
            elif self.conv=='GCNConv':
                self.up_convs.append(gnn.GCNConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2]))
            elif self.conv=='GATConv':
                self.up_convs.append(gnn.GATConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2]))
            else:
                raise NotImplementedError('Invalid convolution selected. Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')
            
        self.reset_parameters()

    def decoder(self, x, data):
        """
        Executes the decoding pass.

        Args:
            x (torch.Tensor): Latent bottleneck tensor [Batch, Bottleneck].
            data (torch_geometric.data.Data): Graph metadata (edge_index, etc.).

        Returns:
            torch.Tensor: Reconstructed node features [Nodes*Batch, Output_Features].
        """
        x = self.act(self.fc_out1(x))
        x = self.act(self.fc_out2(x))
        h = x.reshape(data.num_graphs*self.input_size, self.hidden_channels[-1])
        x = h
        idx = 0
        for layer in self.up_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = layer(x, data.edge_index, data.edge_weight)
            elif self.conv in ['GATConv']:
                x = layer(x, data.edge_index, data.edge_attr)
            if (idx != self.depth - 2):
                x = self.act(x)
            if self.skip:
                x = x + h
            idx += 1
        return x

    def reset_parameters(self):
        """Initializes weights using Kaiming Uniform and sets biases to zero."""
        for conv in self.up_convs:
            conv.reset_parameters()
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_uniform_(param)

    def forward(self, x, data):
        x = self.decoder(x, data)
        return x