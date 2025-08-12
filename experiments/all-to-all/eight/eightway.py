import torch
import torch.distributed as dist
import Linear 
import distributed_matmul as dist_mm
import utils
import time
import torch.profiler

device, slurm_localid, rank, world_size = utils.init_CUDA()    
utils.set_all_seeds(seed=30, rank=rank)

x_dim, y_dim, z_dim = 8, 24, 4
split1, split2 = 4, 2

#x = torch.rand(x_dim//split1, y_dim//split2).to(device)
#y = torch.rand(y_dim//split2, z_dim//split1).to(device).mT.contiguous()
#
#dist_mm.XWT_eightway(x, y, rank=rank, device=device, model_group=None)
#
#x = torch.rand(x_dim//split1, y_dim//split2).to(device)
#y = torch.rand(y_dim//split1, z_dim//split2).to(device)
#
#dist_mm.XW_eightway(x, y, rank=rank, device=device, model_group=None)
#
#x = torch.rand(x_dim//split2, y_dim//split1).to(device).mT.contiguous()
#y = torch.rand(y_dim//split1, z_dim//split2).to(device)
#
#Cij = dist_mm.XTW_eightway(x, y, rank=rank, device=device, model_group=None)
#

# Local dimension of x and y: [1, 2, 8]
#x = torch.rand(1, 6, 4).to(device)
#model = Linear.DistributedXTW(8, 6, 2, rank=rank, device=device).to(device)
## Goal output: [1, 8]
#model(x, model_group=None)

#x = torch.rand(1, 4, 6).to(device)
#model = Linear.DistributedXWT(8, 6, 2, rank=rank, device=device).to(device)
## Local model output and target size: [1, 2, 12]
#model(x, model_group=None)

torch.backends.cuda.matmul.allow_tf32 = False

batch_size = 4
global_xin_dim = 8192
global_hidden_dim = 8192
global_out_dim = 8192
n_parallel = 8

model = Linear.DistributedXWT(global_hidden_dim//2, global_out_dim//4, rank=rank, device=device, n_parallel=n_parallel).to(device)
x = torch.rand(batch_size, global_xin_dim//4, global_hidden_dim//2).to(device)
if rank == 0:
    flops = batch_size * global_xin_dim * global_out_dim * (2 * global_hidden_dim - 1)
    print(f'TFLOPs forward: {flops/1e12}')
    tflops = flops / 1e12 * 3
    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

for i in range(10):
    result = model(x, model_group=None)

start_time = time.time()
n_iters = 250

#with torch.profiler.profile(
#    schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
#    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
#    record_shapes=True,
#    profile_memory=True,
#    with_stack=True
#) as prof:
for i in range(n_iters):
    # prof.step()
    result = model(x, model_group=None)
    loss = loss_fn(result, result*0.1)
    loss.backward()     
    #optimizer.step()
    #optimizer.zero_grad()

torch.cuda.synchronize()

end_time = time.time()

if rank == 0:
    print(loss)
    
if rank == 0:
    total_flops = (tflops + batch_size*global_xin_dim*global_out_dim/1e12) * n_iters
    total_flopss = total_flops / 8.0 / (end_time - start_time)
    print(f'Total time: {end_time - start_time} [s], Total tflops {total_flops}, total tflopss {total_flopss}')

dist.destroy_process_group()

