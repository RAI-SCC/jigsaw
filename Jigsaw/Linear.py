from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from Jigsaw import distributed_matmul as dist_mm


def select_xw_transpose_kernel(n_parallel: int):
    """Return correct xw_t kernel."""
    if n_parallel == 2:
        return dist_mm.xw_transpose_twoway
    elif n_parallel == 4:
        return dist_mm.xw_transpose_fourway
    elif n_parallel == 8:
        return dist_mm.xw_transpose_eightway


def select_xw_kernel(n_parallel: int):
    """Return xw kernel."""
    if n_parallel == 2:
        return dist_mm.xw_twoway
    elif n_parallel == 4:
        return dist_mm.xw_fourway
    elif n_parallel == 8:
        return dist_mm.xw_eightway


def select_x_transpose_w_kernel(n_parallel: int):
    """Return x_tw kernel."""
    if n_parallel == 2:
        return dist_mm.x_transpose_w_twoway
    elif n_parallel == 4:
        return dist_mm.x_transpose_w_fourway
    elif n_parallel == 8:
        return dist_mm.x_transpose_w_eightway


def select_x_transpose_kernel(n_parallel: int):
    """Return x_tw kernel."""
    if n_parallel == 2:
        return dist_mm.x_transpose_twoway
    elif n_parallel == 4:
        return dist_mm.x_transpose_fourway
    elif n_parallel == 8:
        return dist_mm.x_transpose_eightway


class DistributedXWT(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        n_channels: int = 1,
        bias: bool = True,
        stdev: float = None,
        init_method: str = "linear",
        patch_size=None,
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        n_parallel: int
            number of processes it is distributed over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        n_channels : int, optional
            The number of channels (default is 1). Only used for parameter initialization.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        stdev : float, optional
            The standard deviation for weight initialization (default is None).
        init_method : str, optional
            The method for weight initialization, either 'linear' or 'conv' (default is 'linear').
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if init_method == "linear":
            stdev = (
                1.0 / np.sqrt(in_dim * 2) if n_parallel == 4 else 1.0 / np.sqrt(in_dim)
            )  # TODO: multiply by 2 for parallelism = 4, I think
            self.weights = torch.nn.parameter.Parameter(
                torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev)
            )
        elif init_method == "conv":
            stdev = np.sqrt(1.0 / (69 * patch_size))  # TODO generalize to input script
            self.weights = torch.nn.parameter.Parameter(
                torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev)
            )
        elif init_method == "convTranspose":
            stdev = np.sqrt(
                1.0 / (69 * patch_size * 2)
            )  # TODO generalize to input script
            self.weights = torch.nn.parameter.Parameter(
                torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev)
            )
        if bias and n_parallel == 2:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim // 2).uniform_(-stdev, stdev)
            )
        elif bias and n_parallel == 4:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim).uniform_(-stdev, stdev)
            )
        elif bias and n_parallel == 8:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim * 2).uniform_(-stdev, stdev)
            )
        else:
            self.bias = None

        self.device = device
        self.rank = rank
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.XWT = XWT().apply

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
        Cij = self.XWT(
            x,
            self.weights,
            self.bias,
            self.rank,
            self.device,
            self.n_parallel,
            self.model_group,
        )
        return Cij


class XWT(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim).
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        n_parallel : int
            Number of processes to distribute operation across
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
            B, out_dim are global values.
            in_dim is local value.
        """
        xwt = select_xw_transpose_kernel(n_parallel)(x, w, rank, device, model_group)

        # Handle bias term here?
        if (b is not None) and (len(b.shape) > 1):
            b = b.reshape(b.shape[0], 1, 1, -1)

        xwt = xwt if b is None else xwt.add(b)
        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, n_parallel, model_group)
        output: xwt
        """
        x, w, b, rank, device, n_parallel, model_group = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
        """
        Backward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight, bias = ctx.saved_tensors
        rank = ctx.rank
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group

        grad_input = select_xw_kernel(n_parallel)(
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = select_x_transpose_w_kernel(n_parallel)(
            grad_output, input, rank=rank, device=device, model_group=model_group
        )

        if len(grad_output.shape) == 4:
            grad_b = None if bias is None else grad_output.sum(dim=(1, 2))
        elif len(grad_output.shape) == 5:
            grad_b = None if bias is None else grad_output.sum(dim=(1, 2, 3))
        else:
            grad_b = None if bias is None else grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, n_parallel, model_group).
        x : Tensor
            Possibly batched input.
        w : Tensor
            Possibly shared or batched weights.
        b : Tensor or None
            Possibly shared or batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        (
            x_bdim,
            w_bdim,
            b_bdim,
            rank_bdim,
            device_bdim,
            n_parallel_bdim,
            model_group_bdim,
        ) = in_dims

        # We only support vmapping over tensor arguments; non-tensors must not be batched.
        assert rank_bdim is None
        assert device_bdim is None
        assert n_parallel_bdim is None
        assert model_group_bdim is None
        # assert w_bdim is None
        #        assert b_bdim is None

        # Case 1: no vmapping at all → just call original op.
        if x_bdim is None:
            out = XWT.apply(x, w, b, rank, device, n_parallel, model_group)
            return out, None

        # Only support vmap over dim 0 of x.
        if x_bdim != 0:
            raise NotImplementedError(
                f"XWT.vmap only supports vmapping over x in dim 0, got x_bdim={x_bdim}"
            )

        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()
        V = x.shape[0]
        B = x.shape[1]
        rest = w.shape[1:]  # (in_dim, out_dim)

        if len(x.shape) > len(w.shape):
            w = w.reshape(V, 1, *rest)

        out = XWT.apply(x, w, b, rank, device, n_parallel, model_group)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXW(torch.nn.Module):
    """Batched matrix-matrix (XW) multiplication distributed across two processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        n_channels: int = 1,
        bias: bool = False,
        stdev: float = None,
        init_method: str = "linear",
    ) -> None:
        """
        Initialize distributed matrix-matrix multiplication module.

        n_parallel: int
            Number of processes the operation is distributed over
        in_dim: int
            Global input dimension of local weight matrix.
        out_dim: int
            Local output dimension of local weight matrix.
        device: torch.device
            Device to perform computation on.
        rank : int
            Global rank
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        stdev = 1.0 / np.sqrt(in_dim * 2) if n_parallel == 4 else 1.0 / np.sqrt(in_dim)

        self.weights = torch.nn.parameter.Parameter(
            torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev)
        )
        self.bias = (
            torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev))
            if bias
            else None
        )

        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        self.XW = XW().apply

    def forward(self, x: torch.Tensor):
        """
        Forward pass of linear layer (XW) distributed across two processes.

        x: torch.Tensor
            local input tensor of shape (B, x_in, x_out)
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        cij = self.XW(
            x,
            self.weights,
            self.bias,
            self.rank,
            self.device,
            self.n_parallel,
            self.model_group,
        )
        return cij


class XW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XW) multiplication distributed across two processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
            B, x_in are global dimensions.
            in_dim is local dimension.
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
            in_dim is global dimension.
        b : torch.Tensor
            Local bias tensor of shape (out_dim)
        device : Device to perform computation on.
        rank : int
            Global rank
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        xw = select_xw_kernel(n_parallel)(x, w, rank, device, model_group)

        if (b is not None) and (len(b.shape) > 1):
            b = b.reshape(b.shape[0], 1, 1, -1)
        xw = xw if b is None else xw.add(b)
        return xw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, n_parallel, model_group)
        output: xwt
        """
        x, w, b, rank, device, n_parallel, model_group = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        ctx.b = b

    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_output: torch.Tensor):
        """
        Backward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight, bias = ctx.saved_tensors
        rank = ctx.rank
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group

        # TODO: needed add after introduction of multi-head attention value concat
        grad_output = grad_output.contiguous()
        grad_input = select_xw_transpose_kernel(n_parallel)(
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = select_x_transpose_w_kernel(n_parallel)(
            input, grad_output, rank=rank, device=device, model_group=model_group
        )

        if len(grad_output.shape) == 4:
            grad_b = None if bias is None else grad_output.sum(dim=(1, 2))
        else:
            grad_b = None if bias is None else grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, n_parallel, model_group).
        x : Tensor
            Possibly batched input.
        w : Tensor
            Possibly shared or batched weights.
        b : Tensor or None
            Possibly shared or batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        (
            x_bdim,
            w_bdim,
            b_bdim,
            rank_bdim,
            device_bdim,
            n_parallel_bdim,
            model_group_bdim,
        ) = in_dims

        # We only support vmapping over tensor arguments; non-tensors must not be batched.
        assert rank_bdim is None
        assert device_bdim is None
        assert n_parallel_bdim is None
        assert model_group_bdim is None
        # assert w_bdim is None
        assert b_bdim is None

        # Case 1: no vmapping at all → just call original op.
        if x_bdim is None:
            out = XWT.apply(x, w, b, rank, device, n_parallel, model_group)
            return out, None

        # Only support vmap over dim 0 of x.
        if x_bdim != 0:
            raise NotImplementedError(
                f"XWT.vmap only supports vmapping over x in dim 0, got x_bdim={x_bdim}"
            )

        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()
        V = x.shape[0]
        B = x.shape[1]
        rest = w.shape[1:]  # (in_dim, out_dim)
        if len(x.shape) > len(w.shape):
            w = w.reshape(V, 1, *rest)
        out = XW.apply(x, w, b, rank, device, n_parallel, model_group)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXTW(torch.nn.Module):
    """Batched matrix-matrix (XTW) linear layer distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        n_channels: int = 72,
        bias: bool = False,
        init_method: str = "linear",
    ) -> None:
        """
        n_parallel: int
            Number of processes to distribute over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device: torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        n_channels : int, optional
            Number of channels for weight initialization in 'conv' mode. Default is 72.
        bias : bool, optional
            Whether to include a bias term. Default is True.
        init_method : str, optional
            Initialization method for weights ('linear' or 'conv'). Default is 'linear'.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if init_method == "linear":
            stdev = (
                1.0 / np.sqrt(in_dim * 2) if n_parallel == 2 else 1.0 / np.sqrt(in_dim)
            )
            self.weights = torch.nn.parameter.Parameter(
                torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev)
            )
        elif init_method == "conv":
            stdev = np.sqrt(6.0) / (60 * n_channels)  # TODO: 60?
            self.weights = torch.nn.parameter.Parameter(
                torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev)
            )
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim).uniform_(-stdev, stdev)
            )
        else:
            self.bias = None
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        self.XTW = XTW().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        cij = self.XTW(
            x,
            self.weights,
            self.bias,
            self.rank,
            self.device,
            self.n_parallel,
            self.model_group,
        )

        return cij


class XTW(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XTW) multiplication distributed across two processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim).
            Bias vector.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        xtw: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim).
            B, x_in are global values.
            out_dim is local value.
        """
        xtw = select_x_transpose_w_kernel(n_parallel)(x, w, rank, device, model_group)
        if (b is not None) and (len(b.shape) > 1):
            b = b.reshape(b.shape[0], 1, 1, -1)
        xtw = xtw if b is None else xtw.add(b)
        return xtw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, n_parallel, model_group)
        output: xwt
        """

        x, w, b, rank, device, n_parallel, model_group = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group
        ctx.b = b

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None
    ]:
        """
        Backward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output of the layer.

        Returns
        -------
        grad_input: torch.Tensor
            Gradient with respect to the input.
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        input, weight, bias = ctx.saved_tensors
        rank = ctx.rank
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group

        grad_input = select_xw_transpose_kernel(n_parallel)(
            weight, grad_output, rank, device, model_group
        )
        grad_weight = select_xw_kernel(n_parallel)(
            input, grad_output, rank, device, model_group
        )

        if len(grad_output.shape) == 4:
            grad_b = None if bias is None else grad_output.sum(dim=(1, 2))
        else:
            grad_b = None if bias is None else grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_b, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, n_parallel, model_group).
        x : Tensor
            Possibly batched input.
        w : Tensor
            Possibly shared or batched weights.
        b : Tensor or None
            Possibly shared or batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """
        (
            x_bdim,
            w_bdim,
            b_bdim,
            rank_bdim,
            device_bdim,
            n_parallel_bdim,
            model_group_bdim,
        ) = in_dims

        # We only support vmapping over tensor arguments; non-tensors must not be batched.
        assert rank_bdim is None
        assert device_bdim is None
        assert n_parallel_bdim is None
        assert model_group_bdim is None
        # assert w_bdim is None
        # assert b_bdim is None

        # Case 1: no vmapping at all → just call original op.
        if x_bdim is None:
            out = XWT.apply(x, w, b, rank, device, n_parallel, model_group)
            return out, None

        # Only support vmap over dim 0 of x.
        if x_bdim != 0:
            raise NotImplementedError(
                f"XWT.vmap only supports vmapping over x in dim 0, got x_bdim={x_bdim}"
            )

        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()
        V = x.shape[0]
        B = x.shape[1]
        rest = w.shape[1:]  # (in_dim, out_dim)
        
        if len(x.shape) > len(w.shape):
            w = w.reshape(V, 1, *rest)
        out = XTW.apply(x, w, b, rank, device, n_parallel, model_group)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXTranpose(torch.nn.Module):
    """Batched transpose (XT) operation distributed across n processes."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> None:
        """Initialize the Linear class."""
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.transpose = XT().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across two processes.

        Parameters.
        ----------

        X : torch.Tensor
            The input tensor.
        rank : int
            Global rank.
        device : torch.device
            Device to perform computation on.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """
        x_transpose = self.transpose(
            x, self.rank, self.device, self.n_parallel, self.model_group
        )
        return x_transpose


class XT(torch.autograd.Function):
    """Autograd function for batched transpose (XT) operation distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across n processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        x : torch.Tensor
            The input tensor.
        rank : int
            Global rank.
        device : torch.device
            The device on which the computation is performed.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """

        x = select_x_transpose_kernel(n_parallel)(x, rank, device, model_group)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, n_parallel, model_group)
        output: xwt
        """

        x, rank, device, n_parallel, model_group = inputs
        ctx.device = device
        ctx.rank = rank
        ctx.n_parallel = n_parallel
        ctx.model_group = model_group

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        Backward pass of transpose (XT) distributed across two processes.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        grad_output : tensor
            Gradient of the loss with respect to the output of the layer.
        Returns
        -------
        grad_output:
            Transposed gradient of the loss with respect to the input of the layer.
        """
        rank = ctx.rank
        device = ctx.device
        n_parallel = ctx.n_parallel
        model_group = ctx.model_group

        grad_output = select_x_transpose_kernel(n_parallel)(
            grad_output, rank, device, model_group
        )
        return grad_output, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        x = x.contiguous()
        # x under vmap has shape (V, B, x_in, in_dim).
        out = XT.apply(x, rank, device, n_parallel, model_group)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedMatMulXWTranspose(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ) -> None:
        """Initialize the XWT Linear layer sharded over two processes."""
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        super().__init__()
        self.XWT = XWT().apply

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
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
        Cij = self.XWT(
            x, w, b, self.rank, self.device, self.n_parallel, self.model_group
        )

        return Cij


class DistributedMatMulXW(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across two processes."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
    ):
        """Initialize the XWT Linear layer sharded over n processes."""
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel  # TODO: no bias term
        self.model_group = model_group

        super().__init__()
        self.XW = XW().apply

    def forward(self, x, w, b):
        """
        Forward pass of linear layer (XW) distributed across two processes.

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
        Cij = self.XW(
            x, w, b, self.rank, self.device, self.n_parallel, self.model_group
        )

        return Cij
