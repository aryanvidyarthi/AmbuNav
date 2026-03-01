from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Simple graph convolution using adjacency propagation over node dimension."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T], adjacency: [N, N]
        propagated = torch.einsum("nm,bcmt->bcnt", adjacency, x)
        return self.proj(propagated)


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet with dilated temporal convolutions and adaptive adjacency.

    Input:
        x: [B, T, N] or [B, T, N, in_dim]
    Output:
        y: [B, out_dim, N]  # typically out_dim == output_length
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        out_dim: int,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int = 2,
        num_layers: int = 4,
        embed_dim: int = 10,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()

        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            self.filter_convs.append(
                nn.Conv2d(
                    residual_channels,
                    dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=(1, dilation),
                )
            )
            self.gate_convs.append(
                nn.Conv2d(
                    residual_channels,
                    dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=(1, dilation),
                )
            )
            self.residual_convs.append(
                nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
            )
            self.skip_convs.append(
                nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
            )
            self.graph_convs.append(GraphConv(dilation_channels, residual_channels))

        self.end_conv_1 = nn.Conv2d(skip_channels, skip_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(skip_channels, out_dim, kernel_size=(1, 1))

        # Adaptive adjacency parameters.
        self.node_emb_1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.node_emb_2 = nn.Parameter(torch.randn(embed_dim, num_nodes))

    def _adaptive_adjacency(self) -> torch.Tensor:
        adjacency = F.relu(torch.matmul(self.node_emb_1, self.node_emb_2))
        return F.softmax(adjacency, dim=1)

    def _to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(-1)  # [B, T, N, 1]
        if x.ndim != 4:
            msg = "Expected input shape [B, T, N] or [B, T, N, in_dim]"
            raise ValueError(msg)
        return x.permute(0, 3, 2, 1).contiguous()  # [B, in_dim, N, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_model_input(x)
        if x.shape[1] != self.in_dim or x.shape[2] != self.num_nodes:
            msg = (
                f"Input mismatch: expected in_dim={self.in_dim}, num_nodes={self.num_nodes}, "
                f"got in_dim={x.shape[1]}, num_nodes={x.shape[2]}"
            )
            raise ValueError(msg)

        adjacency = self._adaptive_adjacency()

        x = self.start_conv(x)
        skip: torch.Tensor | None = None

        for layer_idx in range(self.num_layers):
            residual = x
            dilation = 2**layer_idx
            left_pad = dilation * (self.kernel_size - 1)

            # Causal padding for temporal convolutions.
            padded = F.pad(x, (left_pad, 0, 0, 0))
            filter_out = torch.tanh(self.filter_convs[layer_idx](padded))
            gate_out = torch.sigmoid(self.gate_convs[layer_idx](padded))
            temporal_out = filter_out * gate_out

            skip_out = self.skip_convs[layer_idx](temporal_out)
            skip = skip_out if skip is None else skip[..., -skip_out.size(3) :] + skip_out

            residual_out = self.residual_convs[layer_idx](temporal_out)
            graph_out = self.graph_convs[layer_idx](temporal_out, adjacency)
            x = residual[..., -residual_out.size(3) :] + residual_out + graph_out

        assert skip is not None
        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)  # [B, out_dim, N, T]

        # Use final temporal step as multi-step forecast head.
        out = out[:, :, :, -1]  # [B, out_dim, N]
        return out
