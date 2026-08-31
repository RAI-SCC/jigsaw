"""Module for the distributed 2-D convolution.

The 1-D and 3-D modules (:class:`jigsaw.modules.conv1d.conv1d.DistributedConv1D`,
:class:`jigsaw.modules.conv3d.conv3d.DistributedConv3D`) mirror this one.
"""

import math
from collections.abc import Sequence
from typing import Callable, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw import hooks
from jigsaw.utils_aggregate import gather_1d_results

from .config import Conv2dConfig
from .functional import Conv2d

# number of spatial axes of a 2-D convolution, ordered (H, W).
_N_SPATIAL = 2


def _pair(x: int | tuple[int, ...]) -> tuple[int, int]:
    """Normalise an ``int`` or 1-/2-element tuple to a 2-element tuple."""
    if isinstance(x, int):
        return (x, x)
    if len(x) == 1:
        return (x[0], x[0])
    return (x[0], x[1])


def _as_spatial_groups(
    spatial_group: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
) -> tuple[Optional[dist.ProcessGroup], ...]:
    """Normalise the ``spatial_group`` argument to a ``(group_h, group_w)`` tuple.

    A single process group keeps the historical behaviour -- the domain is split
    along the width (last) axis only. A sequence is taken axis-by-axis, so
    ``[group_h, group_w]`` splits both spatial axes; ``None`` (or a group of
    size 1) marks an axis that is not split.
    """
    if isinstance(spatial_group, dist.ProcessGroup):
        return (None,) * (_N_SPATIAL - 1) + (spatial_group,)
    groups = tuple(spatial_group)
    if len(groups) != _N_SPATIAL:
        raise ValueError(
            f"spatial_group as a sequence must have {_N_SPATIAL} entries "
            f"(one per spatial axis, ordered (H, W)); got {len(groups)}."
        )
    return groups


class DistributedConv2D(torch.nn.Module):
    """Distributed 2-D convolution."""

    # type-hints
    bias: Optional[Parameter]
    parallelism: int
    config: Conv2dConfig
    bayesian: bool
    spatial_group: Optional[dist.ProcessGroup]
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...]
    channel_group: dist.ProcessGroup
    empty: torch.Tensor
    device: torch.device
    conv2d_apply: Callable[..., torch.Tensor]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: str | tuple[int, int]
    dilation: tuple[int, int]
    channel_comm: str

    def __init__(
        self,
        # --- core conv (required) ---
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        # --- distributed (required) ---
        device: torch.device,
        spatial_group: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
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
        config: Optional[Conv2dConfig] = None,
    ):
        """
        Distributed 2D Convolution.

        input shape of forward pass: (N, Cin, Hin, Win)
        output shape of forward pass: (N, Cout, Hout, Wout)

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
            ``[group_h, group_w]`` splits both spatial axes; ``None`` (or a
            group of size 1) marks an axis that is not split.
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
        config: Optional[Conv2dConfig]
            Configuration and selection strategy of the convolutional layer.
            If None, this will be initialized here.
        """
        super().__init__()
        self.in_channels = in_channels
        # Output channels are sharded across the channel-parallel group.
        self.out_channels = out_channels // channel_group.size()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = _pair(padding)

        self.channel_comm = channel_comm
        self.groups = groups
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
        self.device = device
        self.parallelism = parallelism
        self.dtype = dtype
        self.bayesian = bayesian

        k = np.sqrt(
            1
            / (
                self.in_channels
                * channel_group.size()
                * self.kernel_size[0]
                * self.kernel_size[1]
            )
        )
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
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
            Conv2dConfig(
                self.bayesian,
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

        self.conv2d_apply = Conv2d.apply

    @property
    def spatial_world_size(self) -> int:
        """Number of spatial ranks the parameters are replicated across."""
        return math.prod(g.size() for g in self._spatial_sync_groups)

    def get_parameter_count(self) -> int:
        """Return effective number of local parameters for Distributed Conv2D module."""
        replicas = self.spatial_world_size
        num_param = 0 if self.bias is None else self.bias.numel() // replicas
        return num_param + self.weight.numel() // replicas

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.Conv2d,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> "DistributedConv2D":
        """
        Convert torch.nn.Conv2D to jigsaw based on given process mesh.

        Parameters
        ----------
        model: torch.nn.Conv2D
            Instance of conv2d class that should be converted to jigsaw conv2d.
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups
        device: torch.device
            Device upon which computation happens.
        row_kw: str
            Key for accessing domain process groups.
        col_kw: str
            Key for accessing channel process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.

        Returns
        -------
        model: jigsaw.convolutions.DistributedConv2D
            Instance of jigsaw distributed Conv2D.
            Jigsaw layer contains weights and biases given from torch.nn.Conv2D layer.
        """
        assert model.in_channels % process_mesh_dict[col_kw].size() == 0, (
            "in_features is not divisible by process mesh cols"
        )

        instance = cls(
            in_channels=model.in_channels,
            out_channels=model.out_channels,  # global ?
            kernel_size=cast(tuple[int, ...], model.kernel_size),
            stride=cast(tuple[int, ...], model.stride),
            padding=cast(tuple[int, ...], model.padding),
            dilation=cast(tuple[int, ...], model.dilation),
            groups=model.groups,
            bias=True if model.bias is not None else False,
            spatial_group=process_mesh_dict[row_kw],
            channel_group=process_mesh_dict[col_kw],
            device=device,
            parallelism=process_mesh_dict[model_kw].size(),
            bayesian=False,  # TODO
            dtype=model.weight.dtype,
        )
        # Map weights — clone so the original graph is detached
        in_rank = process_mesh_dict[col_kw].rank()
        in_shard = model.weight.shape[0] // process_mesh_dict[col_kw].size()

        instance.weight.data = (
            model.weight.data[in_rank * in_shard : (in_rank + 1) * in_shard]
            .clone()
            .to(device)
        )

        if model.bias is not None:
            assert instance.bias is not None
            bias_size = model.out_features // process_mesh_dict[col_kw].size()
            instance.bias.data = model.bias.data[
                in_rank * bias_size : (in_rank + 1) * bias_size
            ].clone()

        return instance

    def convert_to_torch(
        self,
        process_mesh_dict: dict,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> torch.nn.Conv2d:
        """
        Convert jigsaw conv2d layer to torch.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.
        """
        weights, bias = self.merge_weights(process_mesh_dict, col_kw)
        conv2d = torch.nn.Conv2d(
            in_channels=cast(int, weights.shape[0]),
            out_channels=cast(int, weights.shape[1] * self.groups),
            kernel_size=cast(tuple[int, int], self.kernel_size),
            padding=cast(tuple[int, int], self.padding),  # not supported
            dilation=cast(tuple[int, int], self.dilation),  # not supported
            groups=self.groups,  # not supported
            bias=True if bias is not None else False,
            padding_mode="zeros",  # hard-coded
        )
        assert isinstance(weights, torch.Tensor)
        conv2d.weights.data = weights
        conv2d.bias = bias

        return conv2d

    def merge_weights(
        self,
        process_mesh_dict: dict,
        col_kw: str = "col",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Merge weights from jigsaw model.

        Parameters will be merged in a way such that they can be transplanted into a
        torch.nn.Conv2D layer natively.

        Parameters
        ----------
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups
        col_kw: str
            Key for accessing col process groups

        Returns
        -------
        global_weights: torch.Tensor
            Merged weights.
        global_bias: Union[torch.Tensor, None]
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
        """Reshard parameters of weights according to a new process group."""
        self.parallelism = to_process_mesh_dict[pm_col_key].size()
        self.config = Conv2dConfig(
            self.bayesian,
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
            f"global weight shape[0] is {weight_shape_global[0]} which is not divisible by {to_process_mesh_dict[pm_col_key].size()}"
        )
        # gather
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
                f"global bias shape is {bias_shape_global} which is not divisible by {to_process_mesh_dict[pm_col_key].size()}"
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

        if isinstance(global_bias, torch.Tensor) and isinstance(
            global_bias, torch.Tensor
        ):
            self.bias = torch.nn.Parameter(
                global_bias[
                    col_rank * col_shard_size : (col_rank + 1) * col_shard_size
                ].contiguous()
            )
            del global_bias

    def _needs_halo(self) -> bool:
        """Whether a spatially-split conv would need neighbour halos.

        With ``stride == kernel_size``, no padding and no dilation every output
        column depends on exactly one spatial rank's input, so the split needs
        no communication. Anything else does.
        """
        zero_pad = self.padding == (0, 0) or self.padding == "valid"
        return not (
            self.kernel_size == self.stride and zero_pad and self.dilation == (1, 1)
        )

    def _forward_local(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass when ``channel_group.size() == 1`` (no channel comm).

        Nothing has to be gathered over the channel dimension, so the custom
        autograd Function buys nothing: native ``conv2d`` plus the spatial-group
        gradient hooks on ``weight`` / ``bias`` produce the same result. This
        fast path only covers the exact-tiling case: no split spatial axis
        (any ``stride`` / ``padding`` / ``dilation``), or a split that tiles
        with no overlap (``stride == kernel_size``, no padding / dilation).
        Halo-needing splits are handled too, but via ``conv2d_apply`` --
        ``forward`` routes them there and they should never reach here.

        The guard below is therefore defensive: running native ``conv2d`` on a
        local shard that needs a halo would silently give wrong values at the
        shard seams, so turn that into a loud error instead.
        """
        if self._spatial_sync_groups and self._needs_halo():
            raise NotImplementedError(
                "_forward_local reached with a halo-needing spatial split; "
                "forward() should have routed this through conv2d_apply."
            )
        conv2d = self.config.conv2d_fn
        if self.bias is not None:
            y = conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        else:
            y = conv2d(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        return cast(torch.Tensor, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the distributed 2-D convolution.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape
            ``(N, Cin / n_channel, Hin, Win / n_spatial)``.

        Returns
        -------
        torch.Tensor
            Local output tensor of shape
            ``(N, Cout / n_channel, Hout, Wout / n_spatial)``.
        """
        # With no channel split the custom autograd Function buys nothing *unless*
        # a spatially-split conv needs a halo exchange; that case still has to go
        # through ``conv2d_apply`` (``conv2d_halo_nway`` handles a channel group
        # of one as the identity gather). ``_spatial_sync_groups`` is non-empty
        # iff some spatial axis is really split -- checking ``spatial_group``
        # alone would miss a height-only split (its width group has size 1).
        if self.channel_group.size() == 1 and not (
            self._spatial_sync_groups and self._needs_halo()
        ):
            return self._forward_local(x)

        out = self.conv2d_apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.in_channels,
            self.spatial_groups,
            self.channel_group,
            self.bayesian,
            self.config,
        )

        return cast(torch.Tensor, out[0])
