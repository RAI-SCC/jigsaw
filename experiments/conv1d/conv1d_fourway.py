import numpy as np
import torch
import torch.nn.functional as F
from Jigsaw import utils
from Jigsaw import Convolutions
from torch.nn.grad import conv1d_input, conv1d_weight


device, _, rank, _ = utils.init_cuda()
batch = 1
in_channels, out_channels = 2, 2  # global channels
L_in = 4 # local
kernel_size = 2
stride = 1
padding, dilation = 0, 1
groups = 1  # auto world size
spatial_group = utils.get_model_group_columns(rank, parallelism=4, n_cols=2)
channel_group = utils.get_model_group_rows(rank, parallelism=4, n_rows=2)
L_out =  int((L_in*2+padding-dilation*(kernel_size-1)-1)/stride+1)

if L_out % 2 == 1 and rank in [2, 3]:
    L_out = L_out // 2 + 1
else:
    L_out = L_out // 2

torch.manual_seed(7)

model_truth = (
    torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, stride=stride)
    .to(torch.float32)
    .to(device)
)  # GROUND TRUTH: can the model learn this kernel?

torch.manual_seed(rank % 2) # Ranks [0,2] and [1,3] should have the same initialization weights

model = (
    Convolutions.DistributedConv1D_complete(
        in_channels,
        out_channels,  # global values
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        rank,
        spatial_group,
        channel_group,
        device,
        n_parallel=4,
        bias=False
    )
    .to(torch.float32)
    .to(device)
)
torch.manual_seed(rank)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if rank in [0, 2]:
    for model_p, truth_p in zip(model.parameters(), model_truth.parameters()):
        model_p.requires_grad_(False)
        model_p.copy_(truth_p[:out_channels//2].detach())
        model_p.requires_grad_(True)
else:
    for model_p, truth_p in zip(model.parameters(), model_truth.parameters()):
        model_p.requires_grad_(False)
        model_p.copy_(truth_p[out_channels//2:].detach())
        model_p.requires_grad_(True)

X = torch.rand([batch, in_channels // 2, L_in]).to(torch.float32).to(device)    
empty = torch.empty(0, device=device, dtype=torch.float32)
X_send = X
X_recv = [torch.zeros(X.shape, dtype=torch.float32).to(device) for i in range(3)]
match rank:
    case 0:
        torch.distributed.all_to_all(
            [empty, X_recv[0], X_recv[1], X_recv[2]],
            [empty, X_send, X_send, X_send],
            async_op=False,
        )
        X_total = torch.concat(
            [
                torch.concat([X, X_recv[0]], dim=1),
                torch.concat([X_recv[1], X_recv[2]], dim=1),
            ],
            dim=2,
        )
    case 1:
        torch.distributed.all_to_all(
            [X_recv[0], empty, X_recv[1], X_recv[2]],
            [X_send, empty, X_send, X_send],
            async_op=False,
        )
        X_total = torch.concat(
            [
                torch.concat([X_recv[0], X], dim=1),
                torch.concat([X_recv[1], X_recv[2]], dim=1),
            ],
            dim=2,
        )
    case 2:
        torch.distributed.all_to_all(
            [X_recv[0], X_recv[1], empty, X_recv[2]],
            [X_send, X_send, empty, X_send],
            async_op=False,
        )
        X_total = torch.concat(
            [
                torch.concat([X_recv[0], X_recv[1]], dim=1),
                torch.concat([X, X_recv[2]], dim=1),
            ],
            dim=2,
        )
    case 3:
        torch.distributed.all_to_all(
            [X_recv[0], X_recv[1], X_recv[2], empty],
            [X_send, X_send, X_send, empty],
            async_op=False,
        )
        X_total = torch.concat(
            [
                torch.concat([X_recv[0], X_recv[1]], dim=1),
                torch.concat([X_recv[2], X], dim=1),
            ],
            dim=2,
        )

X_total.requires_grad_(True)
X.requires_grad_(True)

values = model(X)
y = model_truth(X_total).to(device)

loss_ref = torch.nn.MSELoss(reduction='sum')(y, torch.ones(y.shape).to(device))
loss_parallel = torch.nn.MSELoss(reduction='sum')(values, torch.ones(values.shape).to(device))
loss_ref.backward()
loss_parallel.backward()

grad_output = 2.0 * (y - torch.ones(y.shape).to(device))
g_x_truth = conv1d_input(X_total.shape, model_truth.weight, grad_output, stride=stride, dilation=dilation, padding=padding)

if rank == 0:
    # print(f'x total shape: {X_total.shape}, model weights shape {model_truth.weight.shape}, grad output shape {grad_output.shape}')
    # print(f'reference grad weights: {model_truth.weight.grad}')
    print(f'gx {g_x_truth}')

torch.distributed.destroy_process_group()
