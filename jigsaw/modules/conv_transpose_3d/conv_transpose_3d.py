"""Module for the distributed 3-D transpose convolution.

Mirrors :class:`jigsaw.modules.conv_transpose.conv_transpose.DistributedConvTranspose2D`
for ``F.conv_transpose3d``.
"""

import math
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw import hooks
from jigsaw.utils_aggregate import gather_1d_results

from .config import ConvTranspose3dConfig
from .functional import ConvTranspose3d

# number of spatial axes of a 3-D transpose convolution, ordered (D, H, W).
_N_SPATIAL = 3


def _triple(x: int | tuple[int, ...]) -> tuple[int, int, int]:
    """Convert an ``int`` or 1-/3-element tuple to a 3-element tuple."""
    if isinstance(x, int):
        return (x, x, x)
    if len(x) == 1:
        return (x[0], x[0], x[0])
    return (x[0], x[1], x[2])


def _as_spatial_groups(
    spatial_group: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
) -> tuple[Optional[dist.ProcessGroup], ...]:
    """Normalise ``spatial_group`` to a ``(group_d, group_h, group_w)`` tuple.

    A single process group keeps the historical behaviour -- the domain is split
    along the width (last) axis only. A sequence is taken axis-by-axis, so
    ``[group_d, group_h, group_w]`` splits several spatial axes; ``None`` (or a
    group of size 1) marks an axis that is not split.
    """
    if isinstance(spatial_group, dist.ProcessGroup):
        return (None,) * (_N_SPATIAL - 1) + (spatial_group,)
    groups = tuple(spatial_group)
    if len(groups) != _N_SPATIAL:
        raise ValueError(
            f"spatial_group as a sequence must have {_N_SPATIAL} entries "
            f"(one per spatial axis, ordered (D, H, W)); got {len(groups)}."
        )
    return groups


class DistributedConvTranspose3D(torch.nn.Module):
    """Distributed ConvTranspose 3D."""

    # type hints
    bias: Optional[Parameter]
    parallelism: int
    config: ConvTranspose3dConfig
    bayesian: bool
    spatial_group: Optional[dist.ProcessGroup]
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...]
    channel_group: dist.ProcessGroup
    empty: torch.Tensor
    device: torch.device
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    padding: str | tuple[int, int, int]
    dilation: tuple[int, int, int]

    def __init__(
        self,
        # --- core ConvTranspose (required) ---
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        # --- distributed (required) ---
        device: torch.device,
        spatial_group: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
        channel_group: dist.ProcessGroup,
        # --- core ConvTranspose (optional, pytorch-standard order) ---
        stride: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] = (0,),
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        # --- distributed (optional) ---
        parallelism: int = 4,
        bayesian: bool = False,
        dtype: torch.dtype = torch.float32,
        config: Optional[ConvTranspose3dConfig] = None,
    ):
        """
        Distributed 3D transpose convolution.

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
        spatial_group: dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]
            Domain-parallel ranks. A single process group splits the domain
            along the width axis only (the historical behaviour). A sequence
            ``[group_d, group_h, group_w]`` splits several spatial axes; ``None``
            (or a group of size 1) marks an axis that is not split.
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
        config: Optional[ConvTranspose3dConfig]
            Configuration and selection strategy of the conv transpose layer.
            If None, this will be initialized here.
        """
        super().__init__()

        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)

        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = _triple(padding)

        self.groups = groups
        self.device = device
        self.spatial_groups = _as_spatial_groups(spatial_group)
        # groups that carry a real (>1) split; the parameters are replicated
        # across their product and their gradients summed back over it.
        self._spatial_sync_groups = [
            g for g in self.spatial_groups if g is not None and g.size() > 1
        ]
        # representative single group kept for backward compatibility (the
        # width-axis group, or ``None`` when the width axis is not split).
        self.spatial_group = self.spatial_groups[-1]
        self.channel_group = channel_group
        self.parallelism = parallelism
        self.in_channels = in_channels
        # Output channels are sharded across the channel-parallel group.
        self.out_channels = out_channels // channel_group.size()
        self.bayesian = bayesian

        # Initialize
        k = np.sqrt(
            self.groups
            / (
                self.out_channels
                * channel_group.size()
                * self.kernel_size[0]
                * self.kernel_size[1]
                * self.kernel_size[2]
            )
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.in_channels,
                self.out_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
            .uniform_(-k, k)
            .to(device)
        )
        self.bias = (
            torch.nn.Parameter(
                torch.zeros(self.out_channels).uniform_(-k, k).to(device)
            )
            if bias
            else None
        )
        self.empty = torch.empty(0, device=self.device, dtype=dtype)

        self.config = (
            ConvTranspose3dConfig(
                self.bayesian,
                self.spatial_groups,
                self.channel_group,
                self.bias,
                self.kernel_size,
                self.stride,
                self.device,
            )
            if config is None
            else config
        )

        if self._spatial_sync_groups:
            self.weight.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self._spatial_sync_groups)
            )
            if self.bias is not None:
                self.bias.register_post_accumulate_grad_hook(
                    hooks.make_grad_sync_hook(self._spatial_sync_groups)
                )
        # Broadcast the reference parameters from global rank 0 to every spatial
        # rank. Sequential broadcasts over the per-axis groups (innermost axis
        # first) propagate rank 0's values across the whole spatial grid.
        for p in self.parameters():
            for g in reversed(self._spatial_sync_groups):
                dist.broadcast(p.data, group_src=0, group=g)

        self.convTranspose3d_apply = ConvTranspose3d.apply

    @property
    def spatial_world_size(self) -> int:
        """Number of spatial ranks the parameters are replicated across."""
        return math.prod(g.size() for g in self._spatial_sync_groups)

    def get_parameter_count(self) -> int:
        """Return effective number of local parameters for the module."""
        replicas = self.spatial_world_size
        num_param = 0 if self.bias is None else self.bias.numel() // replicas
        return num_param + self.weight.numel() // replicas

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.ConvTranspose3d,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> "DistributedConvTranspose3D":
        """
        Convert a ``torch.nn.ConvTranspose3d`` to jigsaw based on the given mesh.

        Parameters
        ----------
        model: torch.nn.ConvTranspose3d
            Instance to convert.
        process_mesh_dict: dict
            Process mesh dict containing row, column, and model process groups.
        device: torch.device
            Device upon which computations will occur.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups. Should contain the union of
            the row and column groups.

        Returns
        -------
        model: DistributedConvTranspose3D
            The jigsaw layer, carrying the weights and bias of the input layer.
        """
        assert model.out_channels % process_mesh_dict[col_kw].size() == 0, (
            "out_channels is not divisible by process mesh cols"
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
        # group shards the weight's *output* channel axis (dim 1).
        out_rank = process_mesh_dict[col_kw].rank()
        out_shard = model.weight.shape[1] // process_mesh_dict[col_kw].size()

        instance.weight.data = (
            model.weight.data[:, out_rank * out_shard : (out_rank + 1) * out_shard]
            .clone()
            .to(device)
        )

        if isinstance(model.bias, torch.Tensor) and isinstance(
            instance.bias, torch.Tensor
        ):
            bias_size = model.out_channels // process_mesh_dict[col_kw].size()
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
    ) -> torch.nn.ConvTranspose3d:
        """
        Convert this jigsaw transpose conv layer back to a ``torch.nn.ConvTranspose3d``.

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
        convtranspose3d = torch.nn.ConvTranspose3d(
            in_channels=cast(int, weights.shape[0]),
            out_channels=cast(int, weights.shape[1] * self.groups),
            kernel_size=cast(tuple[int, int, int], self.kernel_size),
            stride=cast(tuple[int, int, int], self.stride),
            padding=cast(tuple[int, int, int], self.padding),
            dilation=cast(tuple[int, int, int], self.dilation),
            groups=self.groups,
            bias=bias is not None,
            padding_mode="zeros",  # hard-coded
        )
        assert isinstance(weights, torch.Tensor)
        convtranspose3d.weight.data = weights
        if isinstance(convtranspose3d.bias, torch.Tensor) and isinstance(
            bias, torch.Tensor
        ):
            convtranspose3d.bias.data = bias

        return convtranspose3d

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
            self.weight, process_mesh_dict, dim=1, pm_key=col_kw
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
        self.config = ConvTranspose3dConfig(
            self.bayesian,
            self.spatial_groups,
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
        )

        from_weight_shape_local = self.weight.shape
        weight_shape_global = (
            from_weight_shape_local[0],
            from_weight_shape_local[1] * from_process_mesh_dict[pm_col_key].size(),
            *from_weight_shape_local[2:],
        )
        to_weight_shape_local = (
            weight_shape_global[0],
            weight_shape_global[1] // to_process_mesh_dict[pm_col_key].size(),
            *weight_shape_global[2:],
        )

        assert weight_shape_global[1] % to_process_mesh_dict[pm_col_key].size() == 0, (
            f"global weight shape[1] is {weight_shape_global[1]} which is not "
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
            to_weight_shape_local[1],
        )
        self.weight = torch.nn.Parameter(
            global_weights[
                :, col_rank * col_shard_size : (col_rank + 1) * col_shard_size, ...
            ].contiguous()
        )
        del global_weights

        if isinstance(global_bias, torch.Tensor):
            col_rank, col_shard_size = (
                to_process_mesh_dict[pm_col_key].rank(),
                to_bias_shape_local,
            )
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the distributed 3-D transpose convolution.

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
        out = self.convTranspose3d_apply(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.in_channels,
            self.spatial_groups,
            self.channel_group,
            self.device,
            self.parallelism,
            self.empty,
            self.bayesian,
            self.config,
        )

        return cast(torch.Tensor, out[0])
