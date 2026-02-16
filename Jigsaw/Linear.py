from typing import Optional, Callable

import numpy as np
import torch
import torch.distributed as dist

from Jigsaw import distributed_matmul as dist_mm
from Jigsaw import distributed_roll as dist_roll
from typing import Tuple

class TransposeSelectionStrategy:
    def __init__(
        self,
        parallelism: int
    ):
        self.x_t = self._transpose_kernel_selection(parallelism)
    
    def _transpose_kernel_selection(parallelism: int) -> Callable:
        """Return x_t kernel."""
        if parallelism == 2:
            return dist_mm.x_transpose_twoway
        elif parallelism == 4:
            return dist_mm.x_transpose_fourway
        elif parallelism == 8:
            return dist_mm.x_transpose_eightway
        
class LinearSelectionStrategy:
    def __init__(
        self,
        parallelism: int,
        bias: torch.Tensor,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
        synchronized_layer: bool,
        linear_layer=True
    ):
        self.mm_kernels = self._mm_kernel_selection(parallelism)
        self.grad_bias = self._grad_bias_selection(parallelism, bias,  synchronized_layer)
        self.add_bias = self._add_bias_selection(bias, bayesian)
        self.reshape_weights = self._reshape_weights_selection(bayesian, linear_layer)
        self.grad_sum = self._sum_grad_weights_selection(linear_layer)
    
    # reshape weights
    def _reshape_weights_selection(self, bayesian: bool, linear_layer: bool) -> Callable:
        """
        Selector function to decide if .
        
        Parameters
        ----------
        bayesian: bool
            Flag to see if the weight reshape is necessary.
        linear_layer: bool
            Flag to see if the module has weights. 
        """
        # If the model is Bayesian and the module is a linear layer, 
        # weight reshape will be necessary.
        if linear_layer and bayesian:
            return self._reshape_weights
        # If it is a distributed matrix operation with no explicit
        # weight sampling, weight reshape is not needed.
        else:
            return self._no_reshape_weights
        
    def _reshape_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        Reshape weights for vmap by unsqueezing in the batch dimension.
        
        Parameters
        ----------
        w: torch.tensor
            model weights.
        """
        w = w.unsqueeze(1)
        return w

    def _no_reshape_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Return weights as-is. Dummy identity function."""
        return w
        
    def _sum_grad_weights_selection(self, linear_layer: bool) -> Callable:
        """Determine if summation over batch dimension is necessary."""
        if linear_layer:
            return self._grad_sum
        else:
            return self._no_grad_sum            
        
    def _no_grad_sum(self, grad_weight: torch.Tensor, batch_dim: torch.Tensor) -> torch.Tensor:
        """Identity function to return gradient weights."""
        return grad_weight
    
    def _grad_sum(self, grad_weight: torch.Tensor, batch_dim: torch.Tensor) -> torch.Tensor:
        """Summation over gradient weights over the batch dimension."""
        return grad_weight.sum(dim=int(batch_dim))

    # Adding xwt + b
    def _add_bias_selection(self, bias: Union[torch.Tensor, None], bayesian: bool) -> Callable:
        """
        Selector function to add bias to weights.
        
        If the bias is None, return identity.
        If the model is bayesian, perform associated bias reshape.
        Else, if bias exists, add bias.
        
        Parameters
        ----------
        bias: torch.Tensor
            Value of the bias term. Not the boolean flag.
        bayesian: bool
            Flag to indicate if the model is bayesian.
        """
        if bias is None:
            return self._return_identity
        elif bayesian:
            return self._add_bias_bayesian
        else:
            return self._add_bias
    
    def _add_bias_bayesian(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Perform reshape for bias and add to x."""
        b = b.unsqueeze(1).unsqueeze(1)
        return x.add(b)
    
    def _add_bias(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add bias to x."""
        return x.add(b)
            
    def _return_identity(self, x, b: None = None) -> torch.Tensor:
        """Return x as identity since bias does not exist."""
        return x
    
    # Reduce bias terms over spatial groups
    def _grad_bias_selection(self, parallelism: int, bias: Union[torch.Tensor, None], synchronized_layer: bool) -> Callable:
        """
        Selector function for calculating gradient of bias.
        
        Returns None when bias term is None.
        If parallelism is 2, all model instances hold different and independent bias parameters.
        Otherwise, reduction over the spatial groups is necessary, since spatial groups will "share" output channels.
        If we have a synchronized layer in the case of SWIN, the bias synchronization will also need to occur across spatial groups. 
        
        Parameters
        ----------
        
        """
        # return identity if bias is None
        if bias is None: 
            return self._no_bias
        # all_reduce bias over space if it is a synchronized linear layer
        elif synchronized_layer:  
            return self._reduce_bias_over_spatial_group
        # all_reduce is not required for 2-way parallel models
        elif parallelism == 2: 
            return self._no_reduce_bias
        # sync required for 4-or-more-way distributed models
        else: 
            return self._reduce_bias_over_spatial_group
        
        
    def _reduce_bias_over_spatial_group(self, grad_output: torch.Tensor, spatial_group: dist.ProcessGroup) -> torch.Tensor:
        """
        Calculate grad wrt bias.
        
        grad_b is equal to the gradient output summed over space and batch dim.
        
        Parameters
        ----------
        grad_output: torch.Tensor
            gradient wrt loss
        spatial_group: dist.ProcessGroup
            process group of ranks distributed across space (share channels).        
        """
        grad_b = grad_output.sum(dim=(-3,-2))
        dist.all_reduce(grad_b, group=spatial_group, op=dist.ReduceOp.SUM)
        return grad_b
    
    def _no_reduce_bias(self, grad_output: torch.Tensor, spatial_group: dist.ProcessGroup = None) -> torch.Tensor:
        """Calculate grad wrt bias when spatial synchronization is not required."""
        return grad_output.sum(dim=(-3,-2))

    def _no_bias(self, grad_output=None, spatial_group=None) -> None:
        """Return identity when bias does not exist."""
        return None
        
    
    def _mm_kernel_selection(self, parallelism: int) -> Callable:
        """
        Select distriuted matrix multiplication kernels. 
        
        Depends on number of model-parallel processes. 
        
        
        Returns
        -------
        List[Callable, Callable, Callable]: pointers to xwt, xw, and xtw kernels.
        """
        xw_t_kernel = self._select_xw_transpose_kernel(parallelism)
        xw_kernel = self._select_xw_kernel(parallelism)
        x_t_w_kernel = self._select_x_transpose_w_kernel(parallelism)
        
        return [xw_t_kernel, xw_kernel, x_t_w_kernel]        

    def _select_xw_transpose_kernel(self, parallelism: int) -> Callable:
        """Return xw_t kernel based on parallelism."""
        match parallelism:
            case 2:
                return dist_mm.xw_transpose_twoway
            case 4:
                return dist_mm.xw_transpose_fourway
            case 8:
                return dist_mm.xw_transpose_eightway


    def _select_xw_kernel(self, parallelism: int) -> Callable:
        """Return xw kernel based on parallelism."""
        match parallelism: 
            case 2:
                return dist_mm.xw_twoway
            case 4:
                return dist_mm.xw_fourway
            case 8:
                return dist_mm.xw_eightway


    def _select_x_transpose_w_kernel(self, parallelism: int) -> Callable:
        """Return x_tw kernel based on parallelism."""
        match parallelism:
            case 2:
                return dist_mm.x_transpose_w_twoway
            case 4:
                return dist_mm.x_transpose_w_fourway
            case 8:
                return dist_mm.x_transpose_w_eightway


class DistributedXWT(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup = None,
        bayesian: bool = False
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the model is distributed over
        model_group: dist.ProcessGroup
            Process group containing all ranks within model-parallel instances
        spatial_group: dist.ProcessGroup
            Process group containing ranks that share channels.
        bayesian: bool
            Boolean flag for whether the model is bayesian
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
        
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev)) if bias else None
        self.device = device
        self.rank = rank
        self.parallelism = parallelism
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(parallelism, self.bias, self.spatial_group, self.bayesian, synchronized_layer=False, linear_layer=True)
        self.XWT = XWT().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
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
            self.selector,
            self.model_group,
            self.spatial_group,
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
        selector: LinearSelectionStrategy,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        bayesian: bool,
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across parallelism processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        W : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim) or (S, in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim) or (S, out_dim).
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        selector : LinearSelectionStrategy
            Instance of LinearSelectionStrategy class.
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        spatial_group: dist.ProcessGroup
            Process group for ranks that share channels and differ across space.
            
        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim) or (S, B, out_dim, in_dim).
        """
        w = selector.reshape_weights(w)
        xwt = selector.mm_kernels[0](x, w, rank, device, model_group)
        xwt = selector.add_bias(xwt, b)
        
        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, selector, spatial_group, bayesian)
        output: xwt
        """
        x, w, b, rank, device, selector, model_group, spatial_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.spatial_group = spatial_group
        ctx.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None, None]:
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
        selector = ctx.selector
        model_group = ctx.model_group
        spatial_group = ctx.spatial_group
        bayesian = ctx.bayesian
        
        weight = selector.reshape_weights(weight)
        
        grad_input = xwt = selector.mm_kernels[1](
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = xwt = selector.mm_kernels[2](
            grad_output, input, rank=rank, device=device, model_group=model_group
        )
        
        grad_weight = selector.grad_sum(grad_weight, ctx.bayesian) # sum along batch dimension

        grad_b = selector.grad_bias(grad_output, spatial_group)

        return grad_input, grad_weight, grad_b, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims: tuple[Optional[int], ...],
        x: torch.Tensor,
        w: torch.Tensor,
        b: Union[torch.Tensor, None],
        rank: int,
        device: torch.device,
        selector: LinearSelectionStrategy,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
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
            is vmapped over. Structure matches (x, w, b, rank, device, selector, model_group, spatial_group, bayesian).
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
        out = XWT.apply(x, w, b, rank, device, selector, model_group, spatial_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXWTSync(torch.nn.Module):
    """Batched matrix-matrix (XWT) multiplication distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup, # Equivalent to spatial group in distributed Windowed attention
        bayesian: bool = False
    ) -> None:
        """
        Initialize the XWT Linear layer sharded over n processes.

        Parameters
        ----------
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes it is distributed over
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        spatial_group: dist.ProcessGroup
            Process group for ranks that share channels and differ across space.
        bayesian : bool
            Flag for whether the model is Bayesian.
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
        
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev, stdev)) if bias else None
        self.device = device
        self.rank = rank
        self.model_group = model_group
        self.sync_group = sync_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(parallelism, self.bias, self.model_group, self.bayesian, synchronized_layer=True, linear_layer=True) 
        
        self.XWT_sync = XWT_sync().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XWT) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim) or (S, B, out_dim, in_dim).
        """
        Cij = self.XWT_sync(
            x,
            self.weights,
            self.bias,
            self.rank,
            self.device,
            self.selector,
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
        selector: LinearSelectionStrategy,
        model_group: dist.ProcessGroup,
        sync_group: dist.ProcessGroup,
        bayesian: bool
    ) -> torch.Tensor:
        """
        Forward pass of distributed linear layer (XWT) synchronzied across domain-but-not-model-parallel processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim) or (S, in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim) or (S, out_dim).
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        selector : LinearSelectionStrategy
            Instance of linear selection strategy class
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        spatial_group: dist.ProcessGroup
            Process group for ranks that share channels and differ across space.

        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, out_dim, in_dim).
        """

        w = selector.reshape_weights(w)
        xwt = selector.mm_kernels[0](x, w, rank, device, model_group)
        xwt = selector.add_bias(xwt, b)

        return xwt

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, selector, model_group, sync_group, bayesian)
        output: xwt
        """
        x, w, b, rank, device, selector, model_group, sync_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.sync_group = sync_group
        ctx.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None], None, None, None, None, None, None]:
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
        mm_kernels = ctx.selector.mm_kernels
        model_group = ctx.model_group
        bayesian = ctx.bayesian
        
        weight = ctx.selector.reshape_weights(weight)
        
        grad_input = mm_kernels[1](
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = mm_kernels[2](
            grad_output, input, rank=rank, device=device, model_group=model_group
        )
        
        grad_b = ctx.selector.grad_bias(grad_output, ctx.sync_group)
        
        # Sum over batch dimension.
        # Batch dim = 0 if not bayesian, = 1 if bayesian
        grad_weight = grad_weight.sum(dim=int(ctx.bayesian))
        
        # Reduce weights across sync group
        dist.all_reduce(grad_weight, group=ctx.sync_group)
        
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
        selector: LinearSelectionStrategy,
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
        # x under vmap has shape (S, B, x_in, in_dim).
        x = x.contiguous()

        out = XWT_sync.apply(
            x, w, b, rank, device, selector, model_group, sync_group, bayesian
        )

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXW(torch.nn.Module):
    """Batched matrix-matrix (XW) multiplication distributed across two processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool = False
    ) -> None:
        """
        Initialize distributed matrix-matrix multiplication module.

        in_dim: int
            Global input dimension of local weight matrix.
        out_dim: int
            Local output dimension of local weight matrix.
        bias: bool
            Boolean flag for bias.
        rank : int
            Local model rank.
        device: torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the operation is distributed over.
        model_group: dist.ProcessGroup
        bayesian: 
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
        self.selector = LinearSelectionStrategy(parallelism, self.bias, self.model_group, self.bayesian, synchronized_layer=False, linear_layer=True)
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
            self.selector,
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
        selector: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
    ) -> torch.Tensor:
        """
        Forward pass of linear layer (XW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
            B, x_in are global dimensions.
            in_dim is local dimension.
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim).
            in_dim is global dimension.
        b : torch.Tensor
            Local bias tensor of shape (out_dim)
        rank : int
            Local model rank
        device : torch.device 
            Device to perform computation on.
        selector: LinearSelectionStrategy
            Instance of LinearSelectionStrategy class.
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        bayesian : bool
            Flag to indicate if the model is bayesian.            

        Returns
        -------
        C: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        w = selector.reshape_weights(w)
        xw = selector.mm_kernels[1](x, w, rank, device, model_group)
        xw = selector.add_bias(xw, b)
        
        return xw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, parallelism, model_group)
        output: xwt
        """
        x, w, b, rank, device, selector, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.b = b
        ctx.bayesian = bayesian

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
        selector = ctx.selector
        model_group = ctx.model_group

        weight = selector.reshape_weights(weight)        
            
        grad_input = selector.mm_kernels[0](
            grad_output, weight, rank=rank, device=device, model_group=model_group
        )
        grad_weight = selector.mm_kernels[2](
            input, grad_output, rank=rank, device=device, model_group=model_group
        )

        grad_weight = selector.grad_sum(grad_weight, ctx.bayesian)
        grad_b = selector.grad_bias(grad_output)

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
        selector: int,
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
            Batched input.
        w : Tensor
            Batched weights.
        b : Tensor or None
            Batched bias.

        Returns
        -------
        output : Tensor
            Batched output with vmapped dimension in front.
        out_dim : Optional[int]
            Index of vmapped dimension in output (0 if batched, None if not).
        """

        x = x.contiguous()
        out = XW.apply(x, w, b, rank, device, selector, model_group, bayesian)

        # out has vmapped dimension at dim 0
        return out, 0


class DistributedXTW(torch.nn.Module):
    """Batched matrix-matrix (XTW) linear layer distributed across n processes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        bayesian: bool = False
    ) -> None:
        """
        in_dim : int
            input dimension of local weight matrix.
        out_dim : int
            output dimension of local weight matrix.
        bias : bool, optional
            Whether to include a bias term. Default is True.
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism : int
            Number of processes to distribute over
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        bayesian : bool
            Flag to indicate if the model is bayesian.
        """
        super().__init__()

        stdev = 1.0 / np.sqrt(in_dim * 2) if parallelism == 4 else 1.0 / np.sqrt(in_dim)
        self.weights = torch.nn.parameter.Parameter(
            torch.zeros(in_dim, out_dim).uniform_(-stdev, stdev)
        )
        
        self.bias = torch.nn.parameter.Parameter(torch.zeros(out_dim).uniform_(-stdev,stdev)) if bias else None
        
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian
        self.selector = SelectionStrategy(parallelism, self.bias, self.model_group, self.bayesian, synchronized_layer=False, linear_layer=True)
        
        self.XTW = XTW().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear layer (XTW) distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, B, x_in, in_dim).

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
            self.selector,
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
        selector: int,
        model_group: dist.ProcessGroup,
        bayesian: bool
    ) -> torch.Tensor:
        """
        Forward pass of distributed linear layer (XTW).

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim) or (S, x_in, in_dim).
        w : torch.Tensor
            Local weight tensor of shape (in_dim, out_dim) or (S, in_dim, out_dim).
        b : torch.Tensor
            Local bias tensor of shape (out_dim) or (S, in_dim, out_dim).
            Bias vector.
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        bayesian: bool
            Flag to indicate if the model is bayesian.

        Returns
        -------
        xtw: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        
        w = selector.reshape_weights(w)
        xtw = selector.mm_kernels[2](x, w, rank, device, model_group)
        xtw = selector.add_bias(xtw, b)
        
        return xtw

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, b, rank, device, selector, model_group, bayesian)
        output: xwt
        """

        x, w, b, rank, device, selector, model_group, bayesian = inputs
        ctx.save_for_backward(x, w, b)
        ctx.rank = rank
        ctx.device = device
        ctx.selector = selector
        ctx.model_group = model_group
        ctx.b = b
        ctx.bayesian = bayesian

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None, None
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
        selector = ctx.selector
        model_group = ctx.model_group
        weight_orig = weight.shape

        weight = selector.reshape_weights(weight)  
            
        grad_input = selector.mm_kernels[0](
            weight, grad_output, rank, device, model_group
        )
        grad_weight = selector.mm_kernels[1](
            input, grad_output, rank, device, model_group
        )

        grad_weight = selector.grad_sum(grad_weight, ctx.bayesian)
        grad_b = selector.grad_bias(grad_output)
        

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
        selector: int,
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
        out = XTW.apply(x, w, b, rank, device, selector, model_group, bayesian)

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
        """
        Initialize the X Transpose class.
        
        Parameters
        ----------
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes to distribute over
        model_group : dist.ProcessGroup
            Process group for an individual model-parallel instance.
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.selector = TransposeSelectionStrategy(self.parallelism)
        
        self.transpose = XT().apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across two processes.

        Parameters.
        ----------

        X : torch.Tensor
            The input tensor.

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
        selector: int,
        model_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of transpose (XT) distributed across n processes.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        rank : int
            Local model rank.
        device : torch.device
            The device on which the computation is performed.
        selector: TransposeSelectionStrategy
            Instance of transpose selection strategy class
        model_group : dist.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        XT: torch.Tensor
            Local transposed tensor.
        """

        x = self.selector.x_t(x, rank, device, model_group)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, rank, device, selector, model_group)
        output: xwt
        """

        x, rank, device, selector, model_group = inputs
        ctx.device = device
        ctx.rank = rank
        ctx.selector = selector
        ctx.model_group = model_group

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
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
        selector = ctx.selector
        model_group = ctx.model_group
        
        grad_output = self.selector.x_t(
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
        selector: int,
        model_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, Optional[int]]:
        x = x.contiguous()
        # x under vmap has shape (V, B, x_in, in_dim).
        out = XT.apply(x, rank, device, selector, model_group)

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
        Forward pass of distributed roll 4 processes.
        
        Function currently only works when rolled across the -2 and -3 dimensions.

        Parameters.
        ----------

        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x_roll: torch.Tensor
            Rolled tensor
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
        shifts: Tuple[int, int],
        dims: Tuple[int, int],
        rank: int,
        device: torch.device,
        empty: torch.Tensor,
        parallelism: int,
        spatial_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of distribued roll across 4 processes.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        shifts: Tuple[int, int]
            Values to shift by
        dims: Tuple[int, int]
            Dimensions in which to perform the rolled operation
        device : torch.device
            The device on which the computation is performed.
        empty: torch.Tensor 
            Empty torch tensor used as buffer
        parallelism: int
            Number of processes the model is distributed over
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
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
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
    """Batched distributed matrix-matrix (XWT) multiplication."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        bayesian: bool
    ) -> None:
        """
        Initialize the Distributed XWT multiplication operation.  

        Parameters
        ----------
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the model is distributed over.
        model_group: dist.ProcessGroup
            Process group containing all ranks within model-parallel instances.
        spatial_group: dist.ProcessGroup
            Process group containing ranks that share channels.
        bayesian: bool
            Boolean flag for whether the model is Bayesian.
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.bayesian = bayesian
        self.selector = LinearSelectionStrategy(parallelism, None, self.model_group, self.bayesian, synchronized_layer=False, linear_layer=False)
        
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
        w : torch.Tensor
            Local weights of shape (out_dim, in_dim) or (S, out_dim, in_dim).
        b : torch Tensor
            Local bias vector of shape (out_dim) or (S, out_dim).
        Returns
        -------
        Cij: torch.Tensor
            Local output tensor of shape (B, x_in, out_dim) or (S, B, x_in, out_dim).
        """
        Cij = self.XWT(
            x, w, b, self.rank, self.device, self.selector, self.model_group, self.spatial_group, self.bayesian
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
        """
        Initialize the distributed XW module.
        
        Parameters
        ----------
        rank : int
            Local model rank.
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes the model is distributed over.
        model_group: dist.ProcessGroup
            Process group containing all ranks within model-parallel instances.
        spatial_group: dist.ProcessGroup
            Process group containing ranks that share channels.
        bayesian: bool
            Boolean flag for whether the model is Bayesian.
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.bayesian = bayesian

        self.selector = LinearSelectionStrategy(parallelism, None, self.model_group, self.bayesian, synchronized_layer=False, linear_layer=False)
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
        """
        Cij = self.XW(
            x, w, b, self.rank, self.device, self.selector, self.model_group, self.bayesian
        )

        return Cij
    
