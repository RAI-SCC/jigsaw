import torch
import torch.distributed as dist
import Linear 
import distributed_matmul as dist_mm
import utils
import time
import yaml
import matplotlib.pyplot as plt

device, slurm_localid, rank, world_size = utils.init_CUDA()    
utils.set_all_seeds(seed=30, rank=0)


with open('config.yaml', 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.SafeLoader)

dataloader = utils.get_dataloader(
    config_dict=config_dict,
    data_path=config_dict['data']['train_data_path'], 
    mode='train', 
    distributed=False, 
    batch_size=config_dict['training']['train_batch'], 
    subset_size=1000, 
    rank=rank
)

batch_size = 4
global_xin_dim = 72
global_hidden_dim = 72
global_out_dim = 72

model = Linear.DistributedXWT_eightway(global_hidden_dim//2, global_out_dim//4, rank=rank, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()


for (input, target) in dataloader:
    input = torch.flatten(input, start_dim=1, end_dim=2).to(device)
    target = torch.flatten(target[0], start_dim=1, end_dim=2).to(device)

    
    result = model(input, model_group=None)
    loss = loss_fn(result, target)
    loss.backward()     
    if rank == 0:
        print(loss)
    optimizer.step()
    optimizer.zero_grad()

# visual
result = (result - target).reshape(1, 720, 360, 36)
plt.pcolormesh(result[0,...,12].detach().cpu(), vmin=-2, vmax=2)
plt.colorbar()
plt.savefig(f'out/rank_{rank}.png')
torch.distributed.destroy_process_group()