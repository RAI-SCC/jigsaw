"""Module definition of ConvTranspositions."""

from typing import Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

from jigsaw.utils_aggregate import gather_1d_results

from .config import ConvTransposeConfig
from .functional import ConvTranspose2d


def _pair(x: int | tuple[int, ...]) -> tuple[int, int]:
    """Convert different types to 2-element tuple."""
    if isinstance(x, int):
        return (x, x)
    if len(x) == 1:
        return (x[0], x[0])
    return (x[0], x[1])


class DistributedConvTranspose2D(torch.nn.Module):
    """Distributed ConvTranspose 2D."""

    # type hints
    bias: Optional[Parameter]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: str | tuple[int, int]
    dilation: tuple[int, int]

    def __init__(
        self,
        # --- core ConvTranspose (required) ---
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        # --- distributed (required) ---
        device: torch.device,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        # --- core ConvTranspose (optional, pytorch-standard order) ---
        stride: int | tuple[int, ...] = 1,
        padding: str | tuple[int, ...] = (0,),
        dilation: tuple[int, ...] = (1,),
        groups: int = 1,
        bias: bool = True,
        # --- distributed (optional) ---
        parallelism: int = 4,
        bayesian: bool = False,
        dtype: torch.dtype = torch.float32,
        config: Optional[ConvTransposeConfig] = None,
    ):
        """
        Distributed ConvTranspose2D.

        input shape of forward pass: (N, Cin, Lin) or (S, N, Cin, Lin)
        output shape of forward pass: (N, Cout, Lout) or (S, N, Cout, Lout)

        Parameters
        ----------
        in_channels: int
            Number of in_channels globally.
        out_channels: int
            Number of in_channels globally.
        kernel_size: int | tuple[int, ...]
            Kernel size.
        device: torch.Device
            Device where the computations will be done.
        spatial_group: dist.ProcessGroup
            Process group for domain-parallel ranks.
        channel_group: dist.ProcessGroup
            Process group for channel-parallel ranks.
        stride: int | tuple[int, ...]
            Stride in the convolution.
        padding: int | tuple[int, ...]
            Padding. Not yet implemented.
        dilation: int | tuple[int, ...]
            Dilation. Not yet implemented.
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
        config: Optional[ConvTransposeConfig]
            Configuration and selection strategy of conv transpose layer.
            If None, this will be initialized here.
        """
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = _pair(padding)

        assert in_channels % 2 == 0, "number of input channels is not divisible by 2"
        assert out_channels % 2 == 0, "number of output channels is not divisible by 2"

        self.groups = groups
        self.device = device
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.parallelism = parallelism
        self.in_channels = in_channels
        self.out_channels = out_channels // 2
        self.bayesian = bayesian

        # Initialize
        k = np.sqrt(
            self.groups
            / (self.out_channels * 2 * self.kernel_size[0] * self.kernel_size[1])
        )
        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.in_channels,
                self.out_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
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
            ConvTransposeConfig(
                self.bayesian,
                self.spatial_group,
                self.channel_group,
                self.bias,
                self.kernel_size,
                self.stride,
                self.device,
            )
            if config is None
            else config
        )

        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.convTranspose2d_apply = ConvTranspose2d.apply

    def get_parameter_count(self) -> int:
        """Return effective number of local parameters for distributed model."""
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
        model: torch.nn.ConvTranspose2d,
        process_mesh_dict: dict,
        device: torch.device,
        row_kw: str = "row",
        col_kw: str = "col",
        model_kw: str = "model",
    ) -> "DistributedConvTranspose2D":
        """
        Convert torch.nn.ConvTranspose2D to jigsaw based on given process mesh.

        Parameters
        ----------
        model: torch.nn.ConvTranspose2d
            Instance of convTranspose2d class that should be converted to jigsaw linear.
        process_mesh_dict: dict
            Process mesh dict that comtains row, column, and model process groups.
        device: torch.device
            Device upon which computations will occur.
        row_kw: str
            Key for accessing row process groups.
        col_kw: str
            Key for accessing col process groups.
        model_kw: str
            Key for accessing model process groups.
            Model process group should contain the Union of the row and column groups.

        Returns
        -------
        model: jigsaw.convolutions.DistributedConvTranspose2D
            Instance of jigsaw distributed ConvTranspose layer.
            Jigsaw layer contains weights and biases given from torch.nn.ConvTranspose2d layer.
        """
        assert model.in_channels % process_mesh_dict[col_kw].size() == 0, (
            "in_features is not divisible by process mesh cols"
        )

        instance = cls(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            dilation=model.dilation,
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
        in_shard = model.in_channels // process_mesh_dict[col_kw].size()

        instance.weight.data = (
            model.weight.data[:, in_rank * in_shard : (in_rank + 1) * in_shard]
            .clone()
            .to(device)
        )

        if isinstance(model.bias, torch.Tensor) and isinstance(
            instance.bias, torch.Tensor
        ):
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
    ) -> torch.nn.ConvTranspose2d:
        """
        Convert jigsaw transpose convolution layer to torch.

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
        convtranspose2d = torch.nn.ConvTranspose2d(
            in_channels=weights.shape[0],
            out_channels=weights.shape[1] * self.groups,
            kernel_size=cast(tuple[int, int], self.kernel_size),
            padding=cast(tuple[int, int], self.padding),  # not supported
            dilation=cast(tuple[int, int], self.dilation),  # not supported
            groups=self.groups,  # not supported
            bias=True if bias is not None else False,
            padding_mode="zeros",  # hard-coded
        )
        convtranspose2d.weights.data = weights
        if isinstance(convtranspose2d.bias, torch.Tensor) and isinstance(
            bias, torch.Tensor
        ):
            convtranspose2d.bias.data = bias

        return convtranspose2d

    def merge_weights(
        self,
        process_mesh_dict: dict,
        col_kw: str = "col",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Merge weights from jigsaw model.

        Parameters will be merged in a way such that they can be transplanted into a
        torch.nn.ConvTranspose2D layer natively.

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
    ) -> None:
        """Reshard parameters of weights according to a new process group."""
        self.parallelism = to_process_mesh_dict[pm_col_key].size()
        self.config = ConvTransposeConfig(
            self.bayesian,
            self.spatial_group,
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
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        cij = self.convTranspose2d_apply(
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
        cij_ = cast(torch.Tensor, cij[0])
        return cij_
