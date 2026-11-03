"""Base class of the distributed convolutions."""

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw import init, parallel
from jigsaw._core.comm import CommOp
from jigsaw._core.shard_repr import shard_repr
from jigsaw.gather import gather_1d_results

from .._common.arg_norm import _as_spatial_groups, _ntuple, _resolve_padding_str
from .._common.channel_comm import resolve_conv_groups
from .config import ConvNdConfig
from .functional import _ConvNd

# concrete torch conv class per spatial rank, for ``convert_to_torch``.
_TORCH_CONV_CLS = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


class _DistributedConvNd(torch.nn.Module):
    """Distributed N-D convolution. Subclasses set ``_n_spatial``."""

    # spatial rank (1, 2 or 3); pinned by the concrete subclass.
    _n_spatial: int

    bias: Parameter | None
    config: ConvNdConfig
    spatial_groups: tuple[dist.ProcessGroup | None, ...]
    channel_group: dist.ProcessGroup
    device: torch.device
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    channel_comm: CommOp | str

    def __init__(
        self,
        # --- core conv (required) ---
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        # --- distributed (required) ---
        device: torch.device,
        spatial_group: "dist.ProcessGroup | Sequence[dist.ProcessGroup | None]",
        channel_group: dist.ProcessGroup,
        # --- core conv (optional, pytorch-standard-order) ---
        stride: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] = (0,),
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        # --- distributed (optional) ---
        dtype: torch.dtype = torch.float32,
        channel_comm: CommOp | str = "auto",
        config: ConvNdConfig | None = None,
    ):
        """Initialize the module.

        Parameters
        ----------
        in_channels : int
            Global number of input channels.
        out_channels : int
            Global number of output channels.
        kernel_size : int | tuple[int, ...]
            Kernel size.
        device : torch.device
            Device of the module.
        spatial_group : dist.ProcessGroup | Sequence[dist.ProcessGroup | None]
            Process groups splitting the spatial axes. A single group splits
            the last axis. In a sequence, ``None`` or a group of size 1 marks an
            axis that is not split.
        channel_group : dist.ProcessGroup
            Process group splitting the channels.
        stride : int | tuple[int, ...]
            Stride in the convolution.
        padding : str | tuple[int, ...]
            Padding, or ``"valid"`` or ``"same"``. ``"same"`` with an odd
            ``dilation * (kernel_size - 1)`` pads asymmetrically and needs an
            unsplit layer.
        dilation : int | tuple[int, ...]
            Dilation.
        groups : int
            Number of groups. Must be ``1`` or a multiple of
            ``channel_group.size()``.
        bias : bool
            Whether to add a bias.
        dtype : torch.dtype
            Data type of the parameters.
        channel_comm : CommOp or str
            Channel communication strategy: ``"all_gather"`` or a
            :class:`~jigsaw.modules.CommOp` (``"auto"``, ``"all_to_all"``,
            ``"isend"``, ``"batch_isend_irecv"``). Ignored if ``config`` is
            given.
        config : ConvNdConfig, optional
            Kernel selection. Defaults to a configuration built from the
            arguments.

        Raises
        ------
        ValueError
            If ``groups`` or ``channel_group.size()`` does not divide the channel
            counts, a tuple argument has the wrong length, or ``padding`` is an
            invalid string or ``"same"`` with a stride.
        NotImplementedError
            If ``groups > 1`` is not a multiple of ``channel_group.size()``, or
            asymmetric ``"same"`` padding is used on a split layer.
        """
        super().__init__()
        to_ntuple = _ntuple(self._n_spatial)

        self.groups = groups
        # ``channel_local``: every channel-parallel rank owns whole conv-groups,
        # so the conv runs on this rank's channel shard with ``groups_local``
        # groups and no channel-group communication. ``groups == 1`` with a real
        # channel group uses the full-channel-gather path.
        self._channel_local, self._groups_local = resolve_conv_groups(
            groups, in_channels, out_channels, channel_group.size()
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Output channels are sharded across the channel-parallel group.
        self.local_out_channels = out_channels // channel_group.size()

        self.kernel_size = to_ntuple(kernel_size)
        self.stride = to_ntuple(stride)
        self.dilation = to_ntuple(dilation)

        # The kernels and the backward primitives take integer padding.
        if isinstance(padding, str):
            self.padding = _resolve_padding_str(
                padding, self.kernel_size, self.stride, self.dilation
            )
        else:
            self.padding = to_ntuple(padding)

        self.channel_comm = channel_comm
        self.spatial_groups = _as_spatial_groups(spatial_group, self._n_spatial)
        self._check_padding(self.spatial_groups, self._channel_local)
        # groups that carry a real (>1) split; the parameters are replicated
        # across their product and their gradients summed back over it.
        self._spatial_sync_groups = [
            g for g in self.spatial_groups if g is not None and g.size() > 1
        ]
        self.channel_group = channel_group
        self.device = device
        self.dtype = dtype

        # Global (unsharded) shapes in torch.nn.ConvNd layout. ``reset_parameters``
        # derives the init distribution from these, so it matches torch.nn.ConvNd.
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
                self.local_out_channels,
                self.in_channels // groups,
                *self.kernel_size,
                device=device,
                dtype=dtype,
            )
        )
        self.bias = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.local_out_channels, device=device, dtype=dtype)
            )
            if bias
            else None
        )
        self.reset_parameters()

        self.config = (
            ConvNdConfig(
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

        for p in self.parameters():
            parallel.mark_replicated(p, self._spatial_sync_groups)
        # Broadcast the reference parameters from global rank 0 to every spatial
        # rank. Sequential broadcasts over the per-axis groups (innermost axis
        # first) propagate rank 0's values across the whole spatial grid.
        for p in self.parameters():
            for g in reversed(self._spatial_sync_groups):
                dist.broadcast(p.data, group_src=0, group=g)

        self._conv_apply: Callable[..., tuple[torch.Tensor, torch.Tensor]] = (
            _ConvNd.apply
        )

    def reset_parameters(self) -> None:
        """Initialize the parameters like ``torch.nn.Conv{1,2,3}d``.

        The fans are computed from the global weight shape.
        """
        init.kaiming_uniform_(
            self.weight, a=math.sqrt(5), global_shape=self.global_weight_shape
        )
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.global_weight_shape)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def _check_padding(
        self,
        spatial_groups: tuple[dist.ProcessGroup | None, ...],
        channel_local: bool,
    ) -> None:
        """Reject asymmetric ``"same"`` padding unless the native conv runs it.

        Raises
        ------
        NotImplementedError
            If ``self.padding`` is ``"same"`` and a spatial axis is split, or the
            channels are split without each rank owning whole conv-groups.
        """
        split = any(g is not None and g.size() > 1 for g in spatial_groups)
        if self.padding == "same" and (split or not channel_local):
            raise NotImplementedError(
                f"padding='same' with kernel_size={self.kernel_size} and "
                f"dilation={self.dilation} pads asymmetrically, which is only "
                "supported without a spatial split and with whole conv-groups per "
                "channel rank; use an odd kernel_size or explicit padding."
            )

    @property
    def spatial_world_size(self) -> int:
        """Return the number of spatial ranks the parameters are replicated over."""
        return math.prod(g.size() for g in self._spatial_sync_groups)

    def get_parameter_count(self) -> int:
        """Return the global number of parameters of this layer.

        Returns
        -------
        int
            Global (unsharded) count of this layer's parameters, including plain
            ``torch.nn`` children; children with ``get_parameter_count`` report
            their own. No communication.

        Notes
        -----
        This method calculates the number of parameters based on the global weight and bias shape calculated at initialization instead of counting the number of elements of the local parameters and scaling according to the respective communication group sizes.
        """
        num_bias_param = (
            0 if self.global_bias_shape is None else math.prod(self.global_bias_shape)
        )
        return num_bias_param + math.prod(self.global_weight_shape)

    def extra_repr(self) -> str:
        """Return the ``torch.nn.Conv{1,2,3}d`` fields and the shard sizes.

        Returns
        -------
        str
            The fields torch shows, then ``spatial`` (one size per spatial axis)
            and ``channel``.
        """
        # Same fields and defaults as torch's ``_ConvNd.extra_repr``.
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return (
            s
            + ", "
            + shard_repr(spatial=self.spatial_groups, channel=self.channel_group)
        )

    @classmethod
    def convert_to_jigsaw(
        cls,
        model: torch.nn.modules.conv._ConvNd,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
        **kwargs: Any,
    ) -> "_DistributedConvNd":
        """Convert a ``torch.nn.Conv{1,2,3}d`` to a distributed convolution.

        Parameters
        ----------
        model : torch.nn.Conv1d | torch.nn.Conv2d | torch.nn.Conv3d
            Instance to convert.
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        device : torch.device
            Device of the converted module.
        row_kw : str
            Key of the spatial process group.
        col_kw : str
            Key of the channel process group.
        model_kw : str
            Key of the model process group.
        **kwargs : Any
            Unused. Accepts the options of the other converters, which
            :func:`jigsaw.convert.convert_to_jigsaw` passes to every layer.

        Returns
        -------
        _DistributedConvNd
            Distributed convolution with the weights of ``model``.

        Raises
        ------
        ValueError
            If the channel group size does not divide the channel counts.
        NotImplementedError
            If ``model.padding_mode`` is not ``"zeros"``.
        """
        if model.padding_mode != "zeros":
            raise NotImplementedError(
                f"padding_mode={model.padding_mode!r} is not supported; distributed "
                "convolutions pad with zeros only."
            )
        instance = cls(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            dilation=model.dilation,
            groups=model.groups,
            bias=model.bias is not None,
            spatial_group=process_mesh_dict[row_kw],
            channel_group=process_mesh_dict[col_kw],
            device=device,
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
        """Convert this jigsaw conv layer back to a ``torch.nn.Conv{1,2,3}d``.

        Parameters
        ----------
        process_mesh_dict : dict
            Process groups keyed by ``row_kw``, ``col_kw`` and ``model_kw``.
        row_kw : str
            Key of the spatial process group.
        col_kw : str
            Key of the channel process group.
        model_kw : str
            Key of the model process group.

        Returns
        -------
        torch.nn.modules.conv._ConvNd
            Convolution with the merged weights.
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Merge the channel-sharded weights and bias into global tensors.

        Parameters
        ----------
        process_mesh_dict : dict
            Process groups keyed by ``col_kw``.
        col_kw : str
            Key of the channel process group.

        Returns
        -------
        global_weights : torch.Tensor
            Merged weights.
        global_bias : torch.Tensor | None
            Merged bias, if any.
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
        """Reshard the weight and bias onto a new process mesh.

        The module is rebound to the new mesh.

        Parameters
        ----------
        from_process_mesh_dict : dict
            Process groups of the current mesh.
        to_process_mesh_dict : dict
            Process groups of the new mesh.
        pm_row_key : str
            Key of the spatial process group.
        pm_col_key : str
            Key of the channel process group.
        pm_model_key : str
            Key of the model process group.

        Raises
        ------
        ValueError
            If the target channel group size does not divide the channel counts.
        NotImplementedError
            If ``groups > 1`` is not a multiple of the target channel group size,
            or asymmetric ``"same"`` padding would run on a split layer.
        """
        # Validate the target mesh before the gather mutates anything.
        to_channel_local, _ = resolve_conv_groups(
            self.groups,
            self.in_channels,
            self.out_channels,
            to_process_mesh_dict[pm_col_key].size(),
        )
        self._check_padding(
            _as_spatial_groups(to_process_mesh_dict[pm_row_key], self._n_spatial),
            to_channel_local,
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
        self.out_channels = weight_shape_global[0]
        self.local_out_channels = self.out_channels // self.channel_group.size()
        self._channel_local, self._groups_local = resolve_conv_groups(
            self.groups,
            self.in_channels,
            self.out_channels,
            self.channel_group.size(),
        )
        self.config = ConvNdConfig(
            self.channel_group,
            self.bias,
            self.kernel_size,
            self.stride,
            self.device,
            self.channel_comm,
            self.groups,
        )

        # The re-sliced parameters are fresh Parameters, so the __init__ marks
        # are gone. Re-mark them over the spatial groups and re-broadcast
        # rank 0's slice across every spatial rank (innermost axis first).
        for p in self.parameters():
            parallel.mark_replicated(p, self._spatial_sync_groups)
        for p in self.parameters():
            for g in reversed(self._spatial_sync_groups):
                dist.broadcast(p.data, group_src=0, group=g)

    def _needs_halo(self) -> bool:
        """Return whether the spatial split requires a halo exchange."""
        zero = (0,) * self._n_spatial
        unit = (1,) * self._n_spatial
        return not (
            self.kernel_size == self.stride
            and self.padding == zero
            and self.dilation == unit
        )

    def _forward_local(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Convolve the local shard with the native convolution.

        Used if no channel communication is needed. Raises if the spatial split
        requires a halo exchange.
        """
        if self._spatial_sync_groups and self._needs_halo():
            raise NotImplementedError(
                "_forward_local reached with a halo-needing spatial split; "
                "forward() should have routed this through _conv_apply."
            )
        # Under torch.vmap the native primitive's own batching rule handles a
        # vmapped weight.
        y = self.config.native_conv_fn(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self._groups_local,
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the distributed convolution.

        Parameters
        ----------
        x : torch.Tensor
            Local input of shape ``(N, Cin / n_channel, *spatial / n_spatial)``.

        Returns
        -------
        torch.Tensor
            Local output of shape ``(N, Cout / n_channel, *spatial' / n_spatial)``.
        """
        # Channel-local convs need no channel communication, so they skip the
        # autograd Function unless a spatial split needs a halo exchange.
        # Sync before the branch: a patchify conv (kernel == stride, no padding)
        # takes ``_forward_local`` even when spatially split, and its replicated
        # weight and bias still need their gradients summed.
        weight, bias = parallel.sync_grad(self.weight, self.bias)
        weight = cast(torch.Tensor, weight)
        if self._channel_local and not (
            self._spatial_sync_groups and self._needs_halo()
        ):
            return self._forward_local(x, weight, bias)

        out = self._conv_apply(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.spatial_groups,
            self.channel_group,
            self.config,
        )

        return out[0]
