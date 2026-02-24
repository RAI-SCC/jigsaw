import torch
import torch.distributed as dist

from Jigsaw import Convolutions, utils

device, _, rank, _ = utils.init_cuda()
batch = 1
in_channels, out_channels = 4, 4  # global
H, W = 4, 4  # local
kernel_size = (2, 2)  # global kernel size
stride = (2, 2)
padding, dilation = (0, 0), (1, 1)
groups = 1

spatial_group = utils.get_model_group_columns(rank, parallelism=4, n_cols=2)
channel_group = utils.get_model_group_rows(rank, parallelism=4, n_rows=2)

H_out = int(
    ((H * 2 + padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
)  # H is not split here
W_out = int(
    ((W * 2 + padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
)

W_out = (
    W_out // 2 + 1 if (W_out % 2 == 1) and (rank in [0, 2]) else W_out // 2
)  # RHS gets more

# print(f'Rank {rank} and W_out value {W_out}')
torch.manual_seed(
    7
)  # for now set to 1 value s.t. all weights across processes are the same


model_truth = (
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=True)
    .to(torch.float32)
    .to(device)
)  # GROUND TRUTH: can the model learn this kernel?
torch.manual_seed(
    rank % 2
)  # Ranks [0,2] and [1,3] should have the same initialization weights

model = (
    Convolutions.DistributedConv2D(
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
        bias=True,
    )
    .to(torch.float32)
    .to(device)
)

if rank in [0, 2]:
    for model_p, truth_p in zip(model.parameters(), model_truth.parameters()):
        model_p.requires_grad_(False)
        model_p.copy_(truth_p[: out_channels // 2].detach())
        model_p.requires_grad_(True)
        # if len(model_p.shape) == 1:
        #     print(f'shape {model_p.shape} model_p  {model_p}, truth sum {truth_p}')
        if rank == 0:
            print(f"truth param shape {truth_p.shape}, model p shape {model_p.shape}")
else:
    for model_p, truth_p in zip(model.parameters(), model_truth.parameters()):
        model_p.requires_grad_(False)
        model_p.copy_(truth_p[out_channels // 2 :].detach())
        model_p.requires_grad_(True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
iter_ = 0

torch.manual_seed(rank)

X = torch.rand([1, in_channels // 2, H, W]).to(torch.float32).to(device)

empty = torch.empty(0, device=device, dtype=torch.float32)
X_recv0 = torch.zeros(X.shape, dtype=torch.float32).to(device)
X_recv1 = torch.zeros(X.shape, dtype=torch.float32).to(device)
X_recv2 = torch.zeros(X.shape, dtype=torch.float32).to(device)
X_send = X

# print(f'Rank {rank} X shape {X_send.shape}, recv0 shape {X_recv0.shape}, recv1 shape {X_recv1.shape}, recv2 shape {X_recv2.shape}')
match rank:
    case 0:
        dist.all_to_all(
            [empty, X_recv0, X_recv1, X_recv2],
            [empty, X_send, X_send, X_send],
            async_op=False,
        )
        # print(X_recv0)
        X_row1 = torch.concat([X, X_recv0], dim=1)
        X_row2 = torch.concat([X_recv1, X_recv2], dim=1)
        X_total = torch.concat([X_row1, X_row2], dim=3)
    case 1:
        dist.all_to_all(
            [X_recv0, empty, X_recv1, X_recv2],
            [X_send, empty, X_send, X_send],
            async_op=False,
        )
        # print(X_recv0)
        X_row1 = torch.concat([X_recv0, X], dim=1)
        X_row2 = torch.concat([X_recv1, X_recv2], dim=1)
        X_total = torch.concat([X_row1, X_row2], dim=3)
    case 2:
        dist.all_to_all(
            [X_recv0, X_recv1, empty, X_recv2],
            [X_send, X_send, empty, X_send],
            async_op=False,
        )
        X_row1 = torch.concat([X_recv0, X_recv1], dim=1)
        X_row2 = torch.concat([X, X_recv2], dim=1)
        X_total = torch.concat([X_row1, X_row2], dim=3)
    case 3:
        dist.all_to_all(
            [X_recv0, X_recv1, X_recv2, empty],
            [X_send, X_send, X_send, empty],
            async_op=False,
        )
        X_row1 = torch.concat([X_recv0, X_recv1], dim=1)
        X_row2 = torch.concat([X_recv2, X], dim=1)
        X_total = torch.concat([X_row1, X_row2], dim=3)

X_total.requires_grad_(True)
X.requires_grad_(True)

# if rank == 0:
#     print(f'Rank {rank} weights {model_truth.weight}, data {X_total}')
# print(f'Rank {rank} x total {X_total}')
y = model_truth(X_total).to(device)
values = model(X)

# if rank == 0:
#     print(f'y output {y}')

# print(f'Rank {rank}, y {y.shape}, values {values.shape}')
loss_parallel = torch.nn.MSELoss(reduction="sum")(
    values, torch.ones(values.shape).to(device)
)
loss_ref = torch.nn.MSELoss(reduction="sum")(y, torch.ones(y.shape).to(device))


loss_parallel.backward()
loss_ref.backward()

weights_grad = model_truth.weight.grad

print(f"truth {y}, parallel output {values}")
# if rank == 0:
# print(f"model weights {model_truth.bias.grad}") #, weight_grad_shape {weights_grad.shape}, sum {weights_grad.sum(dim=(0,2,3))}")
# print(f"weights {X_total.grad}")

torch.distributed.destroy_process_group()
