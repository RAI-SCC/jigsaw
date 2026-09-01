"""Shared base for the distributed 1-/2-/3-D convolutions.

Mirrors PyTorch's ``torch.nn.modules.conv._ConvNd`` -> ``Conv{1,2,3}d`` layering:
:class:`_DistributedConvNd` holds every dimension-independent piece (parameter
init, spatial/channel process-group bookkeeping, gradient sync, weight
merge/reshard, torch <-> jigsaw conversion, the ``forward`` fast-path routing)
and the concrete classes in :mod:`jigsaw.modules.convolution.conv.conv` only pin
``_n_spatial``.
"""

import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw import hooks
from jigsaw.utils_aggregate import gather_1d_results

from .config import ConvNdConfig
from .functional import ConvNd

# concrete torch conv class per spatial rank, for ``convert_to_torch``.
_TORCH_CONV_CLS = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


def _ntuple(n: int) -> Callable[[int | tuple[int, ...]], tuple[int, ...]]:
    """Return a parser that normalises an ``int`` / short tuple to an ``n``-tuple.

    An ``int`` or a length-1 tuple is broadcast to every axis; a longer tuple is
    taken axis-by-axis (and indexes past ``n`` are dropped). A tuple shorter than
    ``n`` but longer than 1 raises ``IndexError`` -- it is under-specified.
    """

    def parse(x: int | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(x, int):
            return (x,) * n
        if len(x) == 1:
            return (x[0],) * n
        return tuple(x[i] for i in range(n))

    return parse


def _as_spatial_groups(
    spatial_group: "dist.ProcessGroup | Sequence[Optional[dist.ProcessGroup]]",
    n_spatial: int,
) -> tuple[Optional[dist.ProcessGroup], ...]:
    """Normalise ``spatial_group`` to an ``n_spatial``-tuple aligned with the axes.

    A single process group keeps the historical behaviour -- the domain is split
    along the last (width) axis only. A sequence is taken axis-by-axis, so a
    full-length sequence splits several spatial axes; ``None`` (or a group of
    size 1) marks an axis that is not split.
    """
    if isinstance(spatial_group, dist.ProcessGroup):
        return (None,) * (n_spatial - 1) + (spatial_group,)
    groups = tuple(spatial_group)
    if len(groups) != n_spatial:
        raise ValueError(
            f"spatial_group as a sequence must have {n_spatial} entries "
            f"(one per spatial axis, innermost last); got {len(groups)}."
        )
    return groups


class _DistributedConvNd(torch.nn.Module):
    """Distributed N-D convolution; concrete subclasses set ``_n_spatial``."""

    # spatial rank (1, 2 or 3); pinned by the concrete subclass.
    _n_spatial: int

    # type-hints
    bias: Optional[Parameter]
    parallelism: int
    config: ConvNdConfig
    bayesian: bool
    spatial_group: Optional[dist.ProcessGroup]
    spatial_groups: tuple[Optional[dist.ProcessGroup], ...]
    channel_group: dist.ProcessGroup
    device: torch.device
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
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
        config: Optional[ConvNdConfig] = None,
    ):
        """
        Distributed N-D convolution.

        input shape of forward pass: ``(N, Cin, *spatial)``
        output shape of forward pass: ``(N, Cout, *spatial')``

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
            along the last (width) axis only (the historical behaviour). A
            sequence is taken axis-by-axis and may split several spatial axes;
            ``None`` (or a group of size 1) marks an axis that is not split.
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
        config: Optional[ConvNdConfig]
            Configuration and selection strategy of the convolutional layer.
            If None, this will be initialized here.
        """
        super().__init__()
        to_ntuple = _ntuple(self._n_spatial)

        self.in_channels = in_channels
        # Output channels are sharded across the channel-parallel group.
        self.out_channels = out_channels // channel_group.size()

        self.kernel_size = to_ntuple(kernel_size)
        self.stride = to_ntuple(stride)
        self.dilation = to_ntuple(dilation)

        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = to_ntuple(padding)

        self.groups = groups
        self.channel_comm = channel_comm
        self.spatial_groups = _as_spatial_groups(spatial_group, self._n_spatial)
        # groups that carry a real (>1) split; the parameters are replicated
        # across their product and their gradients summed back over it.
        self._spatial_sync_groups = [
            g for g in self.spatial_groups if g is not None and g.size() > 1
        ]
        # representative single group kept for backward compatibility (the
        # innermost / width-axis group, or ``None`` when it is not split).
        self.spatial_group = self.spatial_groups[-1]
        self.channel_group = channel_group
        self.device = device
        self.parallelism = parallelism
        self.dtype = dtype
        self.bayesian = bayesian

        k = np.sqrt(
            1 / (self.in_channels * channel_group.size() * math.prod(self.kernel_size))
        )
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(self.out_channels, self.in_channels, *self.kernel_size)
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
            ConvNdConfig(
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

        self._conv_apply: Callable[..., tuple[torch.Tensor, torch.Tensor]] = (
            ConvNd.apply
        )

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
        model: torch.nn.modules.conv._ConvNd,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> "_DistributedConvNd":
        """
        Convert a ``torch.nn.Conv{1,2,3}d`` to jigsaw based on the given mesh.

        Parameters
        ----------
        model: torch.nn.Conv1d | torch.nn.Conv2d | torch.nn.Conv3d
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
        model: _DistributedConvNd
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
    ) -> torch.nn.modules.conv._ConvNd:
        """
        Convert this jigsaw conv layer back to a ``torch.nn.Conv{1,2,3}d``.

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
        torch_cls: Any = _TORCH_CONV_CLS[self._n_spatial]
        conv: torch.nn.modules.conv._ConvNd = torch_cls(
            in_channels=weights.shape[1],
            out_channels=weights.shape[0],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=bias is not None,
            padding_mode="zeros",  # hard-coded
        )
        assert isinstance(weights, torch.Tensor)
        conv.weight.data = weights
        if bias is not None:
            assert conv.bias is not None
            conv.bias.data = bias

        return conv

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
        self.config = ConvNdConfig(
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

    def _needs_halo(self) -> bool:
        """Whether a spatially-split conv would need neighbour halos.

        With ``stride == kernel_size``, no padding and no dilation every output
        position depends on exactly one spatial rank's input, so the split needs
        no communication. Anything else does.
        """
        zero = (0,) * self._n_spatial
        unit = (1,) * self._n_spatial
        zero_pad = self.padding == zero or self.padding == "valid"
        return not (
            self.kernel_size == self.stride and zero_pad and self.dilation == unit
        )

    def _forward_local(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass when ``channel_group.size() == 1`` (no channel comm).

        Nothing has to be gathered over the channel dimension, so the custom
        autograd Function buys nothing: native ``conv{1,2,3}d`` plus the
        spatial-group gradient hooks on ``weight`` / ``bias`` produce the same
        result. This fast path only covers the exact-tiling case: no split
        spatial axis (any ``stride`` / ``padding`` / ``dilation``), or a split
        that tiles with no overlap (``stride == kernel_size``, no padding /
        dilation). Halo-needing splits are handled too, but via ``_conv_apply``
        -- ``forward`` routes them there and they should never reach here.

        The guard below is therefore defensive: running native ``conv`` on a
        local shard that needs a halo would silently give wrong values at the
        shard seams, so turn that into a loud error instead.
        """
        if self._spatial_sync_groups and self._needs_halo():
            raise NotImplementedError(
                "_forward_local reached with a halo-needing spatial split; "
                "forward() should have routed this through _conv_apply."
            )
        conv = self.config.conv_fn
        if self.bias is not None:
            y = conv(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        else:
            y = conv(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        return cast(torch.Tensor, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the distributed N-D convolution.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape ``(N, Cin / n_channel, *spatial / n_spatial)``.

        Returns
        -------
        torch.Tensor
            Local output tensor of shape
            ``(N, Cout / n_channel, *spatial' / n_spatial)``.
        """
        # With no channel split the custom autograd Function buys nothing *unless*
        # a spatially-split conv needs a halo exchange; that case still has to go
        # through ``_conv_apply`` (the halo kernel handles a channel group of one
        # as the identity gather). ``_spatial_sync_groups`` is non-empty iff some
        # spatial axis is really split -- checking ``spatial_group`` alone would
        # miss a non-innermost-only split (its width group has size 1).
        if self.channel_group.size() == 1 and not (
            self._spatial_sync_groups and self._needs_halo()
        ):
            return self._forward_local(x)

        out = self._conv_apply(
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
