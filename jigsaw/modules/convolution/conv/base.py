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

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw import hooks, init
from jigsaw.utils_aggregate import gather_1d_results

from ..common.arg_norm import _as_spatial_groups, _ntuple
from ..common.channel_comm import resolve_conv_groups
from .config import ConvNdConfig
from .functional import ConvNd

# concrete torch conv class per spatial rank, for ``convert_to_torch``.
_TORCH_CONV_CLS = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


class _DistributedConvNd(torch.nn.Module):
    """Distributed N-D convolution; concrete subclasses set ``_n_spatial``."""

    # spatial rank (1, 2 or 3); pinned by the concrete subclass.
    _n_spatial: int

    # type-hints
    bias: Optional[Parameter]
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
            Number of blocked connections from input to output channels
            (PyTorch grouped-convolution semantics). Must be ``1`` or a multiple
            of ``channel_group.size()``; in the latter case each channel-parallel
            rank owns whole conv-groups and the forward/backward run with no
            channel-group communication.
        bias: bool
            Flag for whether bias is present.
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
        # ``channel_local``: every channel-parallel rank owns whole conv-groups,
        # so the conv runs on this rank's channel shard with ``groups_local``
        # groups and no channel-group communication. ``groups == 1`` with a real
        # channel group keeps the historical full-channel-gather path.
        self._channel_local, self._groups_local = resolve_conv_groups(
            groups, in_channels, out_channels, channel_group.size()
        )
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
        self.dtype = dtype
        self.bayesian = bayesian

        # Global (unsharded) weight/bias shape, torch.nn.Conv{1,2,3}d layout
        # ``(C_out, C_in / groups, *kernel)``. ``reset_parameters`` keys the init
        # distribution off these, so a freshly built layer is drawn from the
        # same distribution as the equivalent ``torch.nn.ConvNd`` -- the
        # channel-group shard size only decides which output-channel slice this
        # rank samples.
        self.global_weight_shape = (
            out_channels,
            self.in_channels // groups,
            *self.kernel_size,
        )
        self.global_bias_shape = (out_channels,) if bias else None

        # Grouped conv weight is ``(C_out, C_in / groups, *kernel)`` (PyTorch
        # layout); the channel-parallel group shards the output-channel axis 0.
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(
                self.out_channels,
                self.in_channels // groups,
                *self.kernel_size,
                device=device,
            )
        )
        self.bias = (
            torch.nn.parameter.Parameter(torch.zeros(self.out_channels, device=device))
            if bias
            else None
        )
        self.reset_parameters()

        self.config = (
            ConvNdConfig(
                self.bayesian,
                self.channel_group,
                self.bias,
                self.kernel_size,
                self.stride,
                self.device,
                self.channel_comm,
                self.groups,
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

    def reset_parameters(self) -> None:
        """
        Re-draw ``weight`` (and ``bias``) from the same distribution as ``torch.nn.Conv{1,2,3}d``.

        PyTorch draws ``weight`` from ``kaiming_uniform_(a=sqrt(5))`` and
        ``bias`` from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``, both keyed off the
        *global* ``fan_in = (C_in / groups) * prod(kernel)``. Each
        channel-parallel rank fills only its own output-channel weight/bias
        shard, so the channel-sharded parameter is distributed exactly as the
        non-distributed one would be. (The subsequent broadcast to every
        spatial rank only replicates that draw; it does not change its
        distribution.)

        Override this method in a subclass, or call :mod:`jigsaw.init` helpers
        directly with ``global_shape=self.global_weight_shape``, to install a
        different init scheme.
        """
        init.kaiming_uniform_(
            self.weight, a=math.sqrt(5), global_shape=self.global_weight_shape
        )
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.global_weight_shape)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

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
            # merged weight is ``(C_out, C_in / groups, *kernel)``.
            in_channels=weights.shape[1] * self.groups,
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
        """
        Re-slice the channel-sharded ``weight`` / ``bias`` onto a new process mesh.

        The gathered global parameters are unchanged; only which output-channel
        slice this rank holds moves. The module is also rebound to
        ``to_process_mesh_dict`` -- spatial/channel process groups, the
        channel-local regime, the kernel-selection ``config`` and the
        spatial-group grad-sync hooks -- so a forward/backward pass on the new
        mesh is immediately valid.
        """
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

        if self.bias is not None and isinstance(global_bias, torch.Tensor):
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

        # Rebind to the target mesh so a forward/backward pass on it is valid.
        self.channel_group = to_process_mesh_dict[pm_col_key]
        self.spatial_groups = _as_spatial_groups(
            to_process_mesh_dict[pm_row_key], self._n_spatial
        )
        self._spatial_sync_groups = [
            g for g in self.spatial_groups if g is not None and g.size() > 1
        ]
        self.spatial_group = self.spatial_groups[-1]
        self.out_channels = weight_shape_global[0] // self.channel_group.size()
        self._channel_local, self._groups_local = resolve_conv_groups(
            self.groups,
            self.in_channels,
            weight_shape_global[0],
            self.channel_group.size(),
        )
        self.config = ConvNdConfig(
            self.bayesian,
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
            self.channel_comm,
            self.groups,
        )

        # The re-sliced parameters are fresh Parameters, so the __init__ hooks
        # are gone. Re-register the spatial-group grad sync and re-broadcast
        # rank 0's slice across every spatial rank (innermost axis first).
        if self._spatial_sync_groups:
            self.weight.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self._spatial_sync_groups)
            )
            if self.bias is not None:
                self.bias.register_post_accumulate_grad_hook(
                    hooks.make_grad_sync_hook(self._spatial_sync_groups)
                )
        for p in self.parameters():
            for g in reversed(self._spatial_sync_groups):
                dist.broadcast(p.data, group_src=0, group=g)

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
        """Forward pass when this rank owns whole conv-groups (no channel comm).

        Covers ``channel_group.size() == 1`` and, for a grouped conv, any channel
        group whose size divides ``groups`` -- either way nothing has to be
        gathered over the channel dimension and the custom autograd Function buys
        nothing: native ``conv{1,2,3}d`` (with ``groups_local`` groups) plus the
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
                groups=self._groups_local,
            )
        else:
            y = conv(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self._groups_local,
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
        # When every channel-parallel rank owns whole conv-groups
        # (``_channel_local`` -- always true with no channel split) the custom
        # autograd Function buys nothing: the conv runs on this rank's channel
        # shard with no channel-group communication. The exception is a
        # spatially-split conv that needs a halo exchange; that still goes through
        # ``_conv_apply`` (its kernel treats a group of one, or a group that owns
        # whole conv-groups, as the identity gather). ``_spatial_sync_groups`` is
        # non-empty iff some spatial axis is really split -- checking
        # ``spatial_group`` alone would miss a non-innermost-only split.
        if self._channel_local and not (
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
            self.spatial_groups,
            self.channel_group,
            self.bayesian,
            self.config,
        )

        return cast(torch.Tensor, out[0])
