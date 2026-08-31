"""Module for the distributed 3-D convolution.

Mirrors :class:`jigsaw.modules.conv2d.conv2d.DistributedConv2D` for ``F.conv3d``.
"""

from typing import Callable, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw import hooks
from jigsaw.utils_aggregate import gather_1d_results

from .config import Conv3dConfig
from .functional import Conv3d


def _triple(x: int | tuple[int, ...]) -> tuple[int, int, int]:
    """Normalise an ``int`` or 1-/3-element tuple to a 3-element tuple."""
    if isinstance(x, int):
        return (x, x, x)
    if len(x) == 1:
        return (x[0], x[0], x[0])
    return (x[0], x[1], x[2])


class DistributedConv3D(torch.nn.Module):
    """Distributed 3D Convolution."""

    # type-hints
    bias: Optional[Parameter]
    parallelism: int
    config: Conv3dConfig
    bayesian: bool
    spatial_group: dist.ProcessGroup
    channel_group: dist.ProcessGroup
    empty: torch.Tensor
    device: torch.device
    conv3d_apply: Callable[..., torch.Tensor]
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    padding: str | tuple[int, int, int]
    dilation: tuple[int, int, int]
    channel_comm: str

    def __init__(
        self,
        # --- core conv (required) ---
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        # --- distributed (required) ---
        device: torch.device,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        # --- core conv (optional, pytorch-standard-order) ---
        stride: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] = (0,),
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        # --- distributed (optional) ---
        parallelism: int = 4,
        bayesian: bool = False,
        dtype: torch.dtype = torch.float32,
        channel_comm: str = "auto",
        config: Optional[Conv3dConfig] = None,
    ):
        """
        Distributed 3D Convolution.

        input shape of forward pass: (N, Cin, Din, Hin, Win)
        output shape of forward pass: (N, Cout, Dout, Hout, Wout)

        Parameters
        ----------
        in_channels: int
            Number of in_channels globally.
        out_channels: int
            Number of out_channels globally.
        kernel_size: int | tuple[int, ...]
            Kernel size.
        device: torch.device
            Device where the computations will be done.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        channel_group: dist.ProcessGroup
            Process group for channel-parallel ranks.
        stride: int | tuple[int, ...]
            Stride in the convolution.
        padding: str | tuple[int, ...]
            Padding. String padding is unsupported once a halo exchange is
            needed (``stride != kernel_size``).
        dilation: int | tuple[int, ...]
            Dilation.
        groups: int
            Number of groups. Not yet implemented.
        bias: bool
            Flag for whether bias is present.
        parallelism: int
            Size of model-parallel groups.
        bayesian: bool
            Flag for whether model is vmapped.
        dtype: torch.dtype
            Data type of the convolution.
        channel_comm: str
            Channel-parallel forward exchange: ``"all_gather"``, ``"p2p"``
            (double-buffered batched p2p, overlaps comm/compute, lower memory
            peak) or ``"all_to_all"`` (same ring over masked all-to-all hops,
            NCCL only). ``"auto"`` (default) -> ``"all_to_all"`` on a CUDA
            ``device``, else ``"p2p"``. Ignored when ``config`` is supplied.
        config: Optional[Conv3dConfig]
            Configuration and selection strategy of the convolutional layer.
            If None, this will be initialized here.
        """
        super().__init__()
        self.in_channels = in_channels
        # Output channels are sharded across the channel-parallel group.
        self.out_channels = out_channels // channel_group.size()

        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)

        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = _triple(padding)

        self.groups = groups
        self.channel_comm = channel_comm
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.device = device
        self.parallelism = parallelism
        self.dtype = dtype
        self.empty = torch.empty(0, device=self.device, dtype=self.dtype)
        self.bayesian = bayesian

        k = np.sqrt(
            1
            / (
                self.in_channels
                * channel_group.size()
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
        )
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
            .uniform_(-k, k)
            .to(device)
        )
        self.bias = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.out_channels).uniform_(-k, k).to(device)
            )
            if bias
            else None
        )

        self.config = (
            Conv3dConfig(
                self.bayesian,
                self.spatial_group,
                self.channel_group,
                self.bias,
                self.kernel_size,
                self.stride,
                self.device,
                self.channel_comm,
            )
            if config is None
            else config
        )

        self.weight.register_post_accumulate_grad_hook(
            hooks.make_grad_sync_hook(self.spatial_group)
        )
        if self.bias is not None:
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.conv3d_apply = Conv3d.apply

    def get_parameter_count(self) -> int:
        """Return effective number of local parameters for the module."""
        num_param = (
            0
            if self.bias is None
            else self.bias.numel() // dist.get_world_size(group=self.spatial_group)
        )
        return num_param + self.weight.numel() // dist.get_world_size(
            group=self.spatial_group
        )

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.Conv3d,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> "DistributedConv3D":
        """
        Convert a ``torch.nn.Conv3d`` to jigsaw based on the given process mesh.

        Parameters
        ----------
        model: torch.nn.Conv3d
            Instance to convert.
        process_mesh_dict: dict
            Process mesh dict containing row, column, and model process groups.
        device: torch.device
            Device upon which computation happens.
        row_kw: str
            Key for accessing domain process groups.
        col_kw: str
            Key for accessing channel process groups.
        model_kw: str
            Key for accessing model process groups. Should contain the union of
            the row and column groups.

        Returns
        -------
        model: DistributedConv3D
            The jigsaw layer, carrying the weights and bias of the input layer.
        """
        assert model.in_channels % process_mesh_dict[col_kw].size() == 0, (
            "in_channels is not divisible by process mesh cols"
        )

        instance = cls(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=cast(tuple[int, ...], model.kernel_size),
            stride=cast(tuple[int, ...], model.stride),
            padding=cast(tuple[int, ...], model.padding),
            dilation=cast(tuple[int, ...], model.dilation),
            groups=model.groups,
            bias=model.bias is not None,
            spatial_group=process_mesh_dict[row_kw],
            channel_group=process_mesh_dict[col_kw],
            device=device,
            parallelism=process_mesh_dict[model_kw].size(),
            bayesian=False,  # TODO
            dtype=model.weight.dtype,
        )
        # Map weights — clone so the original graph is detached. The channel
        # group shards the weight's *output* channel axis (dim 0).
        out_rank = process_mesh_dict[col_kw].rank()
        out_shard = model.weight.shape[0] // process_mesh_dict[col_kw].size()

        instance.weight.data = (
            model.weight.data[out_rank * out_shard : (out_rank + 1) * out_shard]
            .clone()
            .to(device)
        )

        if model.bias is not None:
            assert instance.bias is not None
            bias_size = model.bias.shape[0] // process_mesh_dict[col_kw].size()
            instance.bias.data = (
                model.bias.data[out_rank * bias_size : (out_rank + 1) * bias_size]
                .clone()
                .to(device)
            )

        return instance

    def convert_to_torch(
        self,
        process_mesh_dict: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> torch.nn.Conv3d:
        """
        Convert this jigsaw conv3d layer back to a ``torch.nn.Conv3d``.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict containing row, column, and model process groups.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
        """
        weights, bias = self.merge_weights(process_mesh_dict, col_kw)
        conv3d = torch.nn.Conv3d(
            in_channels=cast(int, weights.shape[1]),
            out_channels=cast(int, weights.shape[0]),
            kernel_size=cast(tuple[int, int, int], self.kernel_size),
            stride=cast(tuple[int, int, int], self.stride),
            padding=cast(tuple[int, int, int], self.padding),
            dilation=cast(tuple[int, int, int], self.dilation),
            groups=self.groups,
            bias=bias is not None,
            padding_mode="zeros",  # hard-coded
        )
        assert isinstance(weights, torch.Tensor)
        conv3d.weight.data = weights
        if bias is not None:
            assert conv3d.bias is not None
            conv3d.bias.data = bias

        return conv3d

    def merge_weights(
        self,
        process_mesh_dict: dict,
        col_kw: str = "col",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Merge the channel-sharded weights (and bias) into global tensors.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict containing row, column, and model process groups.
        col_kw: str
            Key for accessing col process groups.

        Returns
        -------
        global_weights: torch.Tensor
            Merged weights.
        global_bias: Optional[torch.Tensor]
            Merged bias, if it exists.
        """
        global_weights = gather_1d_results(
            self.weight, process_mesh_dict, dim=0, pm_key=col_kw
        )
        global_bias = (
            gather_1d_results(self.bias, process_mesh_dict, dim=0, pm_key=col_kw)
            if self.bias is not None
            else None
        )
        return global_weights, global_bias

    def reshard_parameters(
        self,
        from_process_mesh_dict: dict,
        to_process_mesh_dict: dict,
        pm_row_key: str = "row",
        pm_col_key: str = "col",
        pm_model_key: str = "model",
    ) -> None:
        """Reshard the (channel-sharded) weight and bias onto a new process mesh."""
        self.parallelism = to_process_mesh_dict[pm_col_key].size()
        self.config = Conv3dConfig(
            self.bayesian,
            self.spatial_group,
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
            self.channel_comm,
        )

        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0] * from_process_mesh_dict[pm_col_key].size(),
            *from_weight_shape_local[1:],
        )
        to_weight_shape_local = (
            weight_shape_global[0] // to_process_mesh_dict[pm_col_key].size(),
            *weight_shape_global[1:],
        )

        assert weight_shape_global[0] % to_process_mesh_dict[pm_col_key].size() == 0, (
            f"global weight shape[0] is {weight_shape_global[0]} which is not "
            f"divisible by {to_process_mesh_dict[pm_col_key].size()}"
        )
        global_weights, global_bias = self.merge_weights(
            from_process_mesh_dict, pm_col_key
        )

        if self.bias is not None:
            from_bias_shape_local = self.bias.shape
            bias_shape_global = (
                from_bias_shape_local[0] * from_process_mesh_dict[pm_col_key].size()
            )
            to_bias_shape_local = (
                bias_shape_global // to_process_mesh_dict[pm_col_key].size()
            )
            assert bias_shape_global % to_process_mesh_dict[pm_col_key].size() == 0, (
                f"global bias shape is {bias_shape_global} which is not "
                f"divisible by {to_process_mesh_dict[pm_col_key].size()}"
            )
            global_bias = gather_1d_results(
                self.bias, from_process_mesh_dict, dim=0, pm_key=pm_col_key
            )

        col_rank, col_shard_size = (
            to_process_mesh_dict[pm_col_key].rank(),
            to_weight_shape_local[0],
        )
        self.weight = torch.nn.Parameter(
            global_weights[
                col_rank * col_shard_size : (col_rank + 1) * col_shard_size, ...
            ].contiguous()
        )
        del global_weights

        col_rank, col_shard_size = (
            to_process_mesh_dict[pm_col_key].rank(),
            to_bias_shape_local,
        )
        if isinstance(global_bias, torch.Tensor):
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the distributed 3-D convolution.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape
            ``(N, Cin / n_channel, Din, Hin, Win / n_spatial)``.

        Returns
        -------
        torch.Tensor
            Local output tensor of shape
            ``(N, Cout / n_channel, Dout, Hout, Wout / n_spatial)``.
        """
        out = self.conv3d_apply(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.in_channels,
            self.spatial_group,
            self.channel_group,
            self.device,
            self.parallelism,
            self.empty,
            self.bayesian,
            self.config,
        )

        return cast(torch.Tensor, out[0])
