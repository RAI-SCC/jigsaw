# Jigsaw
Code for performing domain- and tensor-parallel training optimized for large image-based data.

## Structure of repository
- jigsaw/modules contains the implementations of the individual modules. Currently supported operations are ```nn.Linear, nn.Conv2d, nn.ConvTranspose2d, F.roll, Attention (with or without ring), and  non-parameterized matrix multiplications```.

## Environment
This is an installable library. Requirements can be installed with:
```
pip install -e .
```
The dev environment (necessary for running tests) can be installed with
```
pip install -e .[dev]
```

## Usage
_jigsaw_ is designed to follow the interface of PyTorch. The model must be defined exclusively via _jigsaw_ layers. i.e.
```
class Linear(torch.nn.Module):
   def __init__(self, global_in_dim, global_out_dim, process_mesh_dict):
      self.lin = jigsaw.DistributedXWT(global_in_dim, global_out_dim, process_mesh_dict)
   def forward(self, x):
      return self.lin(x)
```
When initializing the models, ensure that each model-parallel rank has a different random seed, thus insuring true randomness in the initialization. The DP wrapper will handle weight initialization synchronization across data-parallel ranks. 
```
local_rank = model_group.rank()
set_all_seeds(local_rank)
model = Linear(global_in_dim, global_out_dim, process_mesh_dict)
```

### Process mesh
Each module accepts a ```process_mesh_dict```, where the last two dimensions represent the DTP. Default kw for the dictionary are ``row`` and ``col``. There are two options
1) Training the model in a fully domain--tensor parallel manner. This means that a process mesh of size (4, 2) will have the parameters sharded across 8 processes.
2) Training the model with domain-parallelism across the ``rows`` axis, and with domain--tensor parallelism across the ``col`` axis. See the individual modules for detailed usage.

### Dataloader
As domain parallelism is applied throughout the model training process, the dataloader needs to be written according.
For a model-parallel group:
```
set_all_seeds(0) # constant number for all instances of the model-parallel group
dataloader = torch.nn.DataLoader(...) # Define the dataloader here
# The Dataset must return the correct shard of the data samples.
```
