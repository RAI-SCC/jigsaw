from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from Jigsaw import distributed_matmul as dist_mm
from Jigsaw import distributed_roll as dist_roll
from typing import Tuple


def select_xw_transpose_kernel(parallelism: int):
    """Return correct xw_t kernel."""
    if parallelism == 2:
        return dist_mm.xw_transpose_twoway
    elif parallelism == 4:
        return dist_mm.xw_transpose_fourway
    elif parallelism == 8:
        return dist_mm.xw_transpose_eightway


def select_xw_kernel(parallelism: int):
    """Return xw kernel."""
    if parallelism == 2:
        return dist_mm.xw_twoway
    elif parallelism == 4:
        return dist_mm.xw_fourway
    elif parallelism == 8:
        return dist_mm.xw_eightway


def select_x_transpose_w_kernel(parallelism: int):
    """Return x_tw kernel."""
    if parallelism == 2:
        return dist_mm.x_transpose_w_twoway
    elif parallelism == 4:
        return dist_mm.x_transpose_w_fourway
    elif parallelism == 8:
        return dist_mm.x_transpose_w_eightway


def select_x_transpose_kernel(parallelism: int):
    """Return x_tw kernel."""
    if parallelism == 2:
        return dist_mm.x_transpose_twoway
    elif parallelism == 4:
        return dist_mm.x_transpose_fourway
    elif parallelism == 8:
        return dist_mm.x_transpose_eightway


class DistributedXWT(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bias: bool = True,
        bayesian: bool = False
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        parallelism: int
            number of processes it is distributed over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        stdev : float, optional
            The standard deviation for weight initialization (default is None).
        """
        super().__init__()
        
        stdev = 1.0 / np.sqrt(in_dim * 2)
        self.weights = torch.nn.parameter.Parameter(
            torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev)
        )

        match parallelism:
            case 2:
                out_dim = out_dim // 2
            case 4:
                out_dim = out_dim
            case 8:
                out_dim = out_dim * 2
        
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim).uniform_(-stdev, stdev)
            )
            self.has_bias = True
        else:
            self.bias = None
            self.has_bias = False

        self.device = device
        self.rank = rank
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian
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
            self.parallelism,
            self.model_group,
            self.bayesian
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
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool,
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
        parallelism : int
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
        if bayesian and len(w.shape) < len(x.shape):
            w = w.unsqueeze(1) # expand to include batch dimension
            if b is not None:
                b = b.unsqueeze(1).unsqueeze(1)
            
        xwt = select_xw_transpose_kernel(parallelism)(x, w, rank, device, model_group)

        xwt = xwt if b is None else xwt.add(b)
        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, parallelism, model_group)
        output: xwt
        """
        x, w, b, rank, device, parallelism, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.has_bias = (b is not None)
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.model_group = model_group
        ctx.bayesian = bayesian

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
        parallelism = ctx.parallelism
        model_group = ctx.model_group
        bayesian = ctx.bayesian
        
        weight_orig = weight.shape
        
        if bayesian and len(weight.shape) < len(grad_output.shape):
            weight = weight.unsqueeze(1)

        grad_input = select_xw_kernel(parallelism)(
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = select_x_transpose_w_kernel(parallelism)(
            grad_output, input, rank=rank, device=device, model_group=model_group
        )
        
        if len(grad_weight.shape) > len(weight_orig):
            grad_weight = grad_weight.sum(dim=int(self.bayesian)) # sum along batch dimension

        grad_b = grad_output.sum(dim=(-3, -2)) if ctx.has_bias else None
        
        # TODO: reduce across spatial dimension
        
        # if len(grad_output.shape) == 4:
        #     grad_b = None if bias is None else grad_output.sum(dim=(1, 2))
        # elif len(grad_output.shape) == 5:
        #     grad_b = None if bias is None else grad_output.sum(dim=(1, 2, 3))
        # else:
        #     grad_b = None if bias is None else grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_b, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, parallelism, model_group).
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

        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()
        V = x.shape[0]
        B = x.shape[1]
        rest = w.shape[1:]  # (in_dim, out_dim)


        out = XWT.apply(x, w, b, rank, device, parallelism, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXWTSync(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup,
        bias: bool = True,
        bayesian: bool = False
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        parallelism: int
            number of processes it is distributed over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device : torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        stdev : float, optional
            The standard deviation for weight initialization (default is None).
        """
        super().__init__()
        
        stdev = (
            1.0 / np.sqrt(in_dim * 2) if parallelism > 1 else 1.0 / np.sqrt(in_dim) # TODO CHECK
        )  
        self.weights = torch.nn.parameter.Parameter(
            torch.zeros(out_dim, in_dim).uniform_(-stdev, stdev)
        )

        match parallelism:
            case 2:
                out_dim = out_dim // 2
            case 4:
                out_dim = out_dim
            case 8:
                out_dim = out_dim * 2
                
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.zeros(out_dim).uniform_(-stdev, stdev)
            )
        else:
            self.bias = None

        self.device = device
        self.rank = rank
        self.parallelism = parallelism
        self.model_group = model_group
        self.sync_group = sync_group
        self.bayesian = bayesian
        
        self.XWT_sync = XWT_sync().apply

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
        Cij = self.XWT_sync(
            x,
            self.weights,
            self.bias,
            self.rank,
            self.device,
            self.parallelism,
            self.model_group,
            self.sync_group,
            self.bayesian
        )
        return Cij


class XWT_sync(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup,
        bayesian: bool
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
        parallelism : int
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

        if bayesian and len(w.shape) < len(x.shape):
            w = w.unsqueeze(1) # expand to include batch dimension
            if b is not None:
                b = b.unsqueeze(1).unsqueeze(1)
            
        xwt = select_xw_transpose_kernel(parallelism)(x, w, rank, device, model_group)

        xwt = xwt if b is None else xwt.add(b)

        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, parallelism, model_group)
        output: xwt
        """
        x, w, b, rank, device, parallelism, model_group, sync_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.model_group = model_group
        ctx.sync_group = sync_group
        ctx.bayesian = bayesian
        ctx.has_bias = (b is not None)

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
        parallelism = ctx.parallelism
        model_group = ctx.model_group
        bayesian = ctx.bayesian
        
        weight_shape_orig = weight.shape
        
        if bayesian and len(weight.shape) < len(grad_output.shape):
            weight = weight.unsqueeze(1)

        grad_input = select_xw_kernel(parallelism)(
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = select_x_transpose_w_kernel(parallelism)(
            grad_output, input, rank=rank, device=device, model_group=model_group
        )

        grad_b = grad_output.sum(dim=(-3, -2)) if ctx.has_bias else None
        
        if len(grad_weight.shape) > len(weight_shape_orig):
            grad_weight = grad_weight.sum(dim=int(ctx.bayesian))
        
        dist.all_reduce(grad_weight, group=ctx.sync_group)
        if grad_b is not None:
            dist.all_reduce(grad_b, group=ctx.sync_group)
            
        return grad_input, grad_weight, grad_b, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup,
        bayesian: bool
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, parallelism, model_group).
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
        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()

        out = XWT_sync.apply(
            x, w, b, rank, device, parallelism, model_group, sync_group, bayesian
        )

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
        parallelism: int,
        model_group: dist.ProcessGroup,
        bias: bool = True,
        bayesian: bool = False
    ) -> None:
        """
        Initialize distributed matrix-matrix multiplication module.

        parallelism: int
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

        stdev = 1.0 / np.sqrt(in_dim * 2) if parallelism == 4 else 1.0 / np.sqrt(in_dim)

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
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian

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
            self.parallelism,
            self.model_group,
            self.bayesian
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
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
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
        if bayesian and len(w.shape) < len(x.shape):
            w = w.unsqueeze(1)
            if b is not None:
                b = b.unsqueeze(1).unsqueeze(1)
            
        xw = select_xw_kernel(parallelism)(x, w, rank, device, model_group)

        xw = xw if b is None else xw.add(b)
        return xw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, parallelism, model_group)
        output: xwt
        """
        x, w, b, rank, device, parallelism, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.model_group = model_group
        ctx.b = b
        ctx.bayesian = bayesian
        ctx.has_bias = b is not None

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
        parallelism = ctx.parallelism
        model_group = ctx.model_group
        weight_orig = weight.shape
        
        if ctx.bayesian and len(weight.shape) < len(grad_output.shape):
            weight = weight.unsqueeze(1)
            
        grad_input = select_xw_transpose_kernel(parallelism)(
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = select_x_transpose_w_kernel(parallelism)(
            input, grad_output, rank=rank, device=device, model_group=model_group
        )

        if len(grad_output.shape) > len(weight_orig):
            grad_weight = grad_weight.sum(dim=int(ctx.bayesian)) # sum along batch dimension
        grad_b = grad_output.sum(dim=(-3, -2)) if ctx.has_bias else None

        return grad_input, grad_weight, grad_b, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, parallelism, model_group).
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

        x = x.contiguous()
        out = XW.apply(x, w, b, rank, device, parallelism, model_group, bayesian)

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
        parallelism: int,
        model_group: dist.ProcessGroup,
        bias: bool = True,
        bayesian: bool = False
    ) -> None:
        """
        parallelism: int
            Number of processes to distribute over
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        device: torch.device
            Device to perform computation on.
        rank : int
            Global rank.
        bias : bool, optional
            Whether to include a bias term. Default is True.
        """
        super().__init__()

        stdev = 1.0 / np.sqrt(in_dim * 2) if parallelism == 4 else 1.0 / np.sqrt(in_dim)
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
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian
        
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
            self.parallelism,
            self.model_group,
            self.bayesian
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
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
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
        
        if bayesian and len(w.shape) < len(x.shape):
            w = w.unsqueeze(1)
            if b is not None:
                b = b.unsqueeze(1).unsqueeze(1)
                
        xtw = select_x_transpose_w_kernel(parallelism)(x, w, rank, device, model_group)
        
        xtw = xtw if b is None else xtw.add(b)
        return xtw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, parallelism, model_group)
        output: xwt
        """

        x, w, b, rank, device, parallelism, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.parallelism = parallelism
        ctx.model_group = model_group
        ctx.b = b
        ctx.has_bias = b is not None
        ctx.bayesian

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
        parallelism = ctx.parallelism
        model_group = ctx.model_group
        weight_orig = weight.shape

        if ctx.bayesian and len(weight.shape) < len(x.shape):
            weight = weight.unsqueeze(1)
            
        grad_input = select_xw_transpose_kernel(parallelism)(
            weight, grad_output, rank, device, model_group
        )
        grad_weight = select_xw_kernel(parallelism)(
            input, grad_output, rank, device, model_group
        )

        if len(grad_weight.shape) > len(weight_orig):
            grad_weight = grad_weight.sum(dim=int(ctx.bayesian))
        
        grad_b = grad_output.sum(dim=(-3, -2)) if ctx.has_bias else None
        
        # if len(grad_output.shape) == 4:
        #     grad_b = None if bias is None else grad_output.sum(dim=(1, 2))
        # else:
        #     grad_b = None if bias is None else grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_b, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Define XWT behavior under torch.vmap.

        Parameters
        ----------
        info :
            Metadata object with info.batch_size (size of vmapped dimension).
        in_dims :
            Tuple of Optional[int] describing which dimension of each input
            is vmapped over. Structure matches (x, w, b, rank, device, parallelism, model_group).
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
        # x under vmap has shape (V, B, x_in, in_dim).
        x = x.contiguous()
        out = XTW.apply(x, w, b, rank, device, parallelism, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXTranspose(torch.nn.Module):
    """Batched transpose (XT) operation distributed across n processes."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> None:
        """Initialize the Linear class."""
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
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
            x, self.rank, self.device, self.parallelism, self.model_group
        )
        return x_transpose


class XT(torch.autograd.Function):
    """Autograd function for batched transpose (XT) operation distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        rank: int,
        device: torch.device,
        parallelism: int,
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

        x = select_x_transpose_kernel(parallelism)(x, rank, device, model_group)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, parallelism, model_group)
        output: xwt
        """

        x, rank, device, parallelism, model_group = inputs
        ctx.device = device
        ctx.rank = rank
        ctx.parallelism = parallelism
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
        parallelism = ctx.parallelism
        model_group = ctx.model_group
        
        grad_output = select_x_transpose_kernel(parallelism)(
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
        parallelism: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        x = x.contiguous()
        # x under vmap has shape (V, B, x_in, in_dim).
        out = XT.apply(x, rank, device, parallelism, model_group)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedRoll(torch.nn.Module):
    """Batched transpose (XT) operation distributed across n processes."""

    def __init__(
        self,
        shifts: Tuple[int, int],
        dims: int,
        rank: int,
        device: torch.device,
        parallelism: int,
        spatial_group: dist.ProcessGroup,
    ) -> None:
        """Initialize the Linear class."""
        super().__init__()

        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.spatial_group = spatial_group
        self.shifts = shifts
        self.dims = dims
        self.roll = Roll().apply
        self.empty = torch.empty(0, device=device, dtype=torch.float32)
        if dims is not None:
            print(f"Alert, dims is {dims} in roll; this is not yet implemented")

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
        x_roll = self.roll(
            x,
            self.shifts,
            self.dims,
            self.rank,
            self.device,
            self.empty,
            self.parallelism,
            self.spatial_group,
        )
        return x_roll


class Roll(torch.autograd.Function):
    """Autograd function for batched transpose (XT) operation distributed across n processes."""

    @staticmethod
    def forward(
        x: torch.Tensor,
        shifts,
        dims,
        rank: int,
        device: torch.device,
        empty: torch.Tensor,
        parallelism: int,
        spatial_group: dist.ProcessGroup,
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
        spatial_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        x_roll: torch.Tensor
            Local rolled tensor.
        """

        x = dist_roll.roll2d(x, shifts, dims, rank, device, empty, spatial_group)

        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, shifts, dims, rank, device, empty, parallelism, spatial_group = inputs

        ctx.shifts = shifts
        ctx.rank = rank
        ctx.dims = dims
        ctx.device = device
        ctx.empty = empty
        ctx.spatial_group = spatial_group

    @staticmethod
    def vmap(
        info, in_dims, x, shifts, dims, rank, device, empty, parallelism, spatial_group
    ):
        out = Roll.apply(
            x, shifts, dims, rank, device, empty, parallelism, spatial_group
        )
        return out, 0

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
        shifts = ctx.shifts
        rank, device = ctx.rank, ctx.device
        spatial_group = ctx.spatial_group
        dims = ctx.dims

        grad_output = dist_roll.roll2d(
            grad_output, shifts, dims, rank, device, ctx.empty, spatial_group
        )
        return grad_output, None, None, None, None, None, None, None


class DistributedMatMulXWTranspose(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
    ) -> None:
        """Initialize the XWT Linear layer sharded over two processes."""
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian
        
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
            x, w, b, self.rank, self.device, self.parallelism, self.model_group, self.bayesian
        )

        return Cij


class DistributedMatMulXW(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across two processes."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
    ):
        """Initialize the XWT Linear layer sharded over n processes."""
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian

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
            x, w, b, self.rank, self.device, self.parallelism, self.model_group, self.bayesian
        )

        return Cij


class AverageGradients(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        sync_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()

        self.model_group = model_group
        self.all_reduce = AvgGradients().apply

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
        Cij = self.all_reduce(x, self.model_group)
        return Cij


class AvgGradients(torch.autograd.Function):
    """Autograd function for batched matrix-matrix (XWT) multiplication distributed across n processes."""

    @staticmethod
    def forward(
        ctx: torch.autograd.Function, x: torch.Tensor, model_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """Forward pass of all reduce."""

        ctx.model_group = model_group

        return x

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
        grad_weight: torch.Tensor
            Gradient with respect to the weights.
        grad_bias: torch.Tensor
            Gradient with respect to the bias.
        """
        dist.all_reduce(grad_output, op=dist.ReduceOp.AVG, group=ctx.model_group)

        return grad_output, None
