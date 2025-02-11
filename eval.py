import torch
import numpy as np
import utils
import time
from pathlib import Path
import pandas as pd

def calc_weight(n_lat, cossum=458.36551167):
    """
    Return latitude-weighted values for loss function.

    n_lat: int
            number of patches in the latitude dimension
    
    Returns
    -------
    weight: np.array(float)
        latitude_based weighting factor
    """
    latitude = np.linspace(np.pi/2.0, -np.pi/2.0, n_lat)
    weight = n_lat * np.cos(latitude) / cossum
    return weight

# upper_variable options: [Z, Q, T, U, V]
# surface_variable options: [MSLP, U10, V10, T2M]
# PL options: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
upper_variable_indexing = {'Z': 0, 'Q': 1, 'T': 2, 'U': 3, 'V': 4}
surface_variable_indexing = {'MSL': 0, 'U10': 1, 'V10': 2, 'T2M': 3}
PL_indexing = {1000: 0, 925: 1, 850: 2, 700: 3, 600: 4, 500: 5, 400: 6, 300: 7, 250: 8, 200: 9, 150: 10, 100: 11, 50: 12}


def get_var_names(prepend_variable=['epoch']):
    surface_variables = ['msl', 'u10', 'v10', 't2m']
    pressure_variables = ['z', 'q', 't', 'u', 'v']
    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    var_names = prepend_variable # list
    var_names.extend(surface_variables)
    for pvar in pressure_variables:
        var_names.extend(pvar + str(pl) for pl in pressure_levels)
    return var_names
    
def gen_plots(config_dict, input_data, output, target_data, rank, file_path, epoch, model_parallel=False):
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    if model_parallel:
        input_print, output_print, target_print = input_data.cpu(), output.cpu(), target_data.cpu()
        input_print  = input_print.reshape(xlat,  xlon, 69).permute(2, 0, 1).cpu()
        output_print = output_print.reshape(xlat, xlon, 69).permute(2, 0, 1).cpu()
        target_print = target_print.reshape(xlat, xlon, 69).permute(2, 0, 1).cpu()

        if rank == 0:
            save_path = f"{file_path}/u10_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=1)
            save_path = f"{file_path}/T2M_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=3)
            save_path = f"{file_path}/T850_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=32)
            save_path = f"{file_path}/U850_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=45)
            save_path = f"{file_path}/Z500_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=9)

    else:
        
        input_print  = input_data[0].reshape(xlat, xlon, 69).permute(2, 0, 1)
        output_print = output[0].reshape(xlat, xlon, 69).permute(2, 0, 1)
        target_print = target_data[0].reshape(xlat, xlon, 69).permute(2, 0, 1)

        save_path = f"{file_path}/u10_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=1)
        save_path = f"{file_path}/T2M_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=3)
        save_path = f"{file_path}/T850_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=32)
        save_path = f"{file_path}/U850_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=45)
        save_path = f"{file_path}/Z500_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=9)

def get_validation_mse(config_dict, model, dataloader, device, rank, epoch, file_path, mse_file_path='rmse.csv', acc_file_path='acc.csv', model_parallel=True, world_size=4, groups=None, node_groups=None):
    if config_dict['model']['model_parallel']:
        mse_file_path = f'rmse_{rank}.csv'
        acc_file_path = f'acc_{rank}.csv'
    loss = torch.zeros(1).to(device)

    rollout_time = config_dict['model']['latent_rollout']
    dt = config_dict['data']['dt']
    rollout_time = (np.arange(rollout_time) + 1) * dt
    var_names = get_var_names(prepend_variable=['epoch', 'rollout_time'])
    
    rmse, rmse_persistence = {}, {}
    acc, acc_persistence = {}, {}
    for rollout in rollout_time:
        rmse[rollout], rmse_persistence[rollout] = {}, {}
        acc[rollout], acc_persistence[rollout] = {}, {}
        rmse[rollout]['epoch'], acc[rollout]['epoch'] = epoch, epoch
        rmse_persistence[rollout]['epoch'], acc_persistence[rollout]['epoch'] = -1, -1
        rmse[rollout]['rollout_time'], rmse_persistence[rollout]['rollout_time'] = rollout, rollout
        acc[rollout]['rollout_time'], acc[rollout]['rollout_time'] = rollout, rollout
        
        # each df has [epoch, rollout_time] as first two values
        for var in var_names[1:]:
            rmse[rollout][var] = torch.tensor([0], device=device)
            acc[rollout][var]  = torch.tensor([0], device=device)
            rmse_persistence[rollout][var] = torch.tensor([0],device=device)
            acc_persistence[rollout][var]  = torch.tensor([0],device=device)
            
    total_samples = torch.tensor([0]).to(device)
    start_validation_time = time.perf_counter()
    area_weights = torch.tensor(calc_weight(721)).to(torch.float32)[:config_dict['data']['xlat']].to(device)
    surface_weights  = torch.tensor([1.5, 0.77, 0.77, 3.0]) * 2
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.77, 0.77]).unsqueeze(1).expand(5, 13).flatten()
    weights_loss = torch.concat([surface_weights, pressure_weights]).view(1, 69).to(device)
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

    index_offset = 2 if 'rollout_time' in var_names else 1
    
    loss_fn = torch.nn.MSELoss()
    buffer  = torch.zeros(config_dict['training']['valid_batch'], xlat, xlon//2, 36).to(device)
    buffer2 = torch.zeros(config_dict['training']['valid_batch'], xlat, xlon//2, 36).to(device)

    for batch, data in enumerate(dataloader):
        for step, rollout in enumerate(rollout_time):
            if not config_dict['model']['model_parallel']:
                input_data  = data[0].to(device)
                target_data = data[1][step].to(device)
            else:
                
                
                if config_dict['model_parallel']['split'] == 2:
                    buffer.zero_()
                    input_data_handle, buffer = utils.send_data_twoway(data[0], batch, buffer, rank, device, groups, config_dict['model_parallel']['alt_sender'])
                    buffer2.zero_()
                    target_data_handle, buffer2 = utils.send_data_twoway(data[1][step], batch, buffer2, rank, device, groups,config_dict['model_parallel']['alt_sender'])
                elif config_dict['model_parallel']['split'] == 4:
                    input_data = data[0].to(device)
                else:
                    buffer.zero_()
                    input_data = utils.send_data_fourway(data[1], buffer, rank, device)
             
                input_data_handle.wait()
                input_data = buffer
            output = model(input_data, rollout_=step+1)
            
            if model_parallel:
                if config_dict['model_parallel']['split'] == 2:
                    #buffer2 = torch.zeros(config_dict['training']['valid_batch'], 36, xlat, xlon//2).to(device)
                    target_data_handle.wait()
                    target_data = buffer2
                elif config_dict['model_parallel']['split'] == 4:
                    target_data = data[1].permute(0, 2, 3, 1).to(device)
                else:
                    buffer.zero_()
                    target_data = utils.send_data_fourway(data[1], buffer, rank, device)
                target_data = target_data.flatten(start_dim=1, end_dim=2)

                loss_   = loss_fn(output, target_data)
                
                loss[0] = loss[0] + loss_.item()

                output = output.reshape(output.shape[0], xlat, xlon//2, 36)
                target_data = target_data.reshape(target_data.shape[0], xlat, xlon//2, 36)
                
                for i, var in enumerate(var_names):
                    if i > index_offset-1:
                        # Only test on surface variables and pressure heights 850, 500 due to compute time
                        if rank in [0, 2]: # first half of variables
                            if i in (np.array([0, 1, 2, 3, 6, 9, 19, 22]) + index_offset):
                                rmse[rollout][var] = rmse[rollout][var] + calc_mse_single_variable_parallel(config_dict, target_data, output, device, area_weights, index=(i-index_offset), norm=True, norm_offset=0)
                                acc[rollout][var]  = acc[rollout][var] + calc_acc_single_variable_parallel(config_dict, target_data, output, device, area_weights, index=(i-index_offset), norm=True, norm_offset=0, rank=rank)
                        else:
                            if i in (np.array([32, 35, 45, 48, 58, 61]) + index_offset):
                                norm_offset = 32
                                rmse[rollout][var] = rmse[rollout][var] + calc_mse_single_variable_parallel(config_dict, target_data, output, device, area_weights, index=(i-index_offset - norm_offset), norm=True, norm_offset=norm_offset)
                                acc[rollout][var]  = acc[rollout][var] + calc_acc_single_variable_parallel(config_dict, target_data, output, device, area_weights, index=(i-index_offset - norm_offset), norm=True, norm_offset=norm_offset, rank=rank)
    
            else:
                input_data = input_data.permute(0, 2, 3, 1) # Permute input data for mse calculation
                target_data = target_data.permute(0, 2, 3, 1).contiguous()
                loss_fn = torch.nn.MSELoss()

                if config_dict['training']['area_weighted']:
                    loss[0] = loss[0] + (loss_fn(torch.einsum('bxyz, x->bxyz', output, area_weights)*weights_loss, torch.einsum('bxyz, x->bxyz', target_data, area_weights)*weights_loss)).item()
                else:
                    loss[0] = loss[0] + (loss_fn(output*weights_loss, target_data*weights_loss)).item()
                # calculate rmse for all variables
                for i, var in enumerate(var_names):
                    if i > index_offset-1:
                        # Only test on surface variables and pressure heights 850, 500
                        # due to saving time 
                        if i in (np.array([0, 1, 2, 3, 6, 9, 19, 22, 32, 35, 45, 48, 58, 61]) + index_offset):
                            rmse[rollout][var] = rmse[rollout][var] + calc_mse_single_variable(config_dict, target_data, output, device, area_weights, index=(i-index_offset), norm=True)
                            acc[rollout][var]  = acc[rollout][var] + calc_acc_single_variable(config_dict, target_data, output, device, area_weights, index=(i-index_offset), norm=True)
                        else:
                            rmse[rollout][var] = rmse[rollout][var] + 0.
                            acc[rollout][var]  = acc[rollout][var]  + 0.

                #if epoch == 0:
                #    for i, var in enumerate(var_names):
                #        if i > 0:
                #            if i in [1, 2, 3, 4, 7, 10, 20, 23, 33, 36, 46, 49, 59, 62]:
                #                rmse_persistence[rollout][var] = rmse[rollout][var] + calc_mse_single_variable(config_dict, target_data, input_data, device, weights, index=(i-1), norm=True)
                #                acc_persistence[rollout][var]  = acc[rollout][var] + calc_acc_single_variable(config_dict, target_data, input_data, device, weights, index=(i-1), norm=True)
                #            else:
                #                rmse[rollout][var] = rmse[rollout][var] + 0.
                #                acc[rollout][var]  = acc[rollout][var] + 0.

        total_samples[0] = total_samples[0] + input_data.shape[0]
        torch.backends.cuda.matmul.allow_tf32 = True
        if batch == 0:
            if model_parallel  and rank < 4: # only needs to happen once
                # Gather input data
                # input data has shape [nvar, lat, lon]
                input_data, output, target_data = input_data[0].contiguous(), output[0].contiguous(), target_data[0].contiguous()
                
                input_tensor_gather = torch.zeros(4, *input_data.shape).to(device)
                torch.distributed.all_gather_into_tensor(input_tensor_gather, input_data, group=node_groups)
                
                top_row = torch.cat([input_tensor_gather[0], input_tensor_gather[1]], dim=2)
                bottom_row = torch.cat([input_tensor_gather[2], input_tensor_gather[3]], dim=2)
                
                # left/right side shape: [1, 720, 720, 72]
                input_data_gathered = torch.cat([top_row, bottom_row], dim=1)[:,:,:69]
                
                # gather first sample to rank 0
                output_tensor_gather = torch.zeros(4, *output.shape).to(device)
                
                torch.distributed.all_gather_into_tensor(output_tensor_gather, output, group=node_groups)
                
                top_row2 = torch.cat([output_tensor_gather[0], output_tensor_gather[1]], dim=2)
                bottom_row2 = torch.cat([output_tensor_gather[2], output_tensor_gather[3]], dim=2)
                # left/right side shape: [1, 720, 720, 72]
                output_gathered = torch.cat([top_row2, bottom_row2], dim=1)[:,:,:69] 
                
                # Gather target data. Currently split between 4 devices as well.
                target_tensor_gather = torch.zeros(4, *target_data.shape).to(device)
                target_data = target_data.contiguous()
                torch.distributed.all_gather_into_tensor(target_tensor_gather, target_data, group=node_groups)
                
                top_row = torch.cat([target_tensor_gather[0], target_tensor_gather[1]], dim=2)
                bottom_row = torch.cat([target_tensor_gather[2], target_tensor_gather[3]], dim=2)
                # left/right side shape: [1, 720, 720, 72]
                target_data_gathered = torch.cat([top_row, bottom_row], dim=1)[:,:,:69]
                
                if rank == 0:
                    gen_plots(config_dict, input_data_gathered, output_gathered, target_data_gathered, rank, file_path, epoch, model_parallel=model_parallel)    
            elif rank == 0:
                
                print(f"rank {rank}, input data shape {input_data.shape}")
                gen_plots(config_dict, input_data, output, target_data, rank, file_path, epoch, model_parallel=model_parallel)
        
    if not config_dict['model']['model_parallel']:
        torch.distributed.all_reduce(total_samples)
        torch.distributed.all_reduce(loss)
        loss /= world_size # average the loss for world size
        loss /= len(dataloader)
    else:
        total_samples = [len(dataloader.dataset)]
        torch.distributed.all_reduce(loss)
        loss /= world_size
    
    print(f"total samples {total_samples}")
        
    for variable in var_names[index_offset:]:
        for rollout in rollout_time:
            if config_dict['model']['model_parallel']: # when paralle, need to reduce among groups
                torch.distributed.all_reduce(rmse[rollout][variable], group=groups[rank%2]) # Adding MSE of two ranks together
                torch.distributed.all_reduce(acc[rollout][variable], group=groups[rank%2], op=torch.distributed.ReduceOp.AVG)
            #if epoch == 0:
            #    torch.distributed.all_reduce(rmse_persistence[rollout][variable])
            #    torch.distributed.all_reduce(acc_persistence[rollout][variable])
            else:
                torch.distributed.all_reduce(rmse[rollout][variable])
                torch.distributed.all_reduce(acc[rollout][variable])
    
    end_validation_time = time.perf_counter()

    if rank == 0:
        print(f"Validation time [min]: {(end_validation_time - start_validation_time)/60}, validation loss {loss.item()}")

        for variable in var_names[index_offset:]:
            for rollout in rollout_time:
                rmse[rollout][variable] = torch.sqrt(rmse[rollout][variable]/total_samples[0])
                acc[rollout][variable]  = acc[rollout][variable] / total_samples[0]
                #if epoch == 0:
                #    rmse_persistence[rollout][variable] = torch.sqrt(rmse_persistence[rollout][variable] / total_samples[0])
                #    acc_persistence[rollout][variable]  = acc_persistence[rollout][variable] / total_samples[0]

                if variable[0] == 'q':
                    rmse[rollout][variable] = rmse[rollout][variable]*1000
                    rmse_persistence[rollout][variable] = rmse_persistence[rollout][variable]*1000
                
    # write to file
    if rank == 0:
        rmse_file, acc_file = Path(mse_file_path), Path(acc_file_path)

        for rollout in rollout_time:
            rmse_values = {key: float(value) for key, value in rmse[rollout].items()}
            acc_values  = {key: float(value) for key, value in acc[rollout].items()}
            #if epoch == 0:
            #    rmse_persistence_values = {key: float(value) for key, value in rmse_persistence[rollout].items()}
            #    acc_persistence_values  = {key: float(value) for key, value in acc_persistence[rollout].items()}
            #    df_rmse = pd.DataFrame([rmse_persistence_values, rmse_values])
            #    df_acc  = pd.DataFrame([acc_persistence_values, acc_values])
            #else:
            df_rmse = pd.DataFrame([rmse_values])
            df_acc  = pd.DataFrame([acc_values])

            if rmse_file.is_file():
                df_rmse.to_csv(mse_file_path, mode='a', header=False)
            else:
                df_rmse.to_csv(mse_file_path)

            if acc_file.is_file():
                df_acc.to_csv(acc_file_path, mode='a', header=False)
            else:
                df_acc.to_csv(acc_file_path)
    return loss.item()

def load_averages():
    static_slevel = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/surface_zarr.npy')
    static_plevel = np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/pressure_zarr.npy')
    mean_slevel   = torch.tensor(static_slevel[0].reshape(4))
    std_slevel    = torch.tensor(static_slevel[1].reshape(4))
    mean_plevel   = torch.tensor(static_plevel[0].reshape(65))
    std_plevel    = torch.tensor(static_plevel[1].reshape(65))

    mean = torch.concatenate([mean_slevel, mean_plevel], axis=0)
    std  = torch.concatenate([std_slevel, std_plevel], axis=0)
    return mean, std
    
def get_climatology():
    climatology_plevel = torch.tensor(np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/pressure_climatology.npy')).reshape(65, 721, 1440)
    climatology_slevel = torch.tensor(np.load('/hkfs/work/workspace/scratch/ke4365-pgw/PanguWeather/constant_masks/surface_climatology.npy'))

    climatology = torch.concatenate([climatology_slevel, climatology_plevel], axis=0).permute(1, 2, 0)
    # climatology [721, 1440, variables]
    return climatology

def calc_mse_single_variable(config_dict, target, result, device, weights, index=0, norm=False):
    if config_dict is None:
        n_lon = 1440
        n_lat = 721
    else:
        n_lon = config_dict['data']['xlon']
        n_lat = config_dict['data']['xlat']

    divisor = torch.sqrt(torch.tensor([n_lon*n_lat])).to(device)

    _, std = load_averages()
    std    = std.to(device)

    std_multiplier = std[index] if norm else 1.0
    
    # MSE = ((target - result) * stdev) ** 2
    if len(target.shape) == 4:
        mean_squared_error = ((target[:, :, :, index] - result[:, :, :, index]) * 
                std_multiplier.flatten() / divisor
            )**2
    else:
        mean_squared_error = ((target - result) * std_multiplier / divisor)**2

    mean_squared_error = torch.einsum('i, aij->ai', weights, mean_squared_error).sum(dim=1)
    mean_squared_error = mean_squared_error.flatten().sum() 
    
    return mean_squared_error.item()
       

def calc_mse_single_variable_parallel(config_dict, target, result, device, weights, index=0, norm=False, norm_offset=0):
    if config_dict is None:
        n_lon = 1440
        n_lat = 721
    else:
        n_lon = config_dict['data']['xlon']
        n_lat = config_dict['data']['xlat']
    # per process: target has shape [n_patches, nlat, nlon//2, nvar/2]
    target_ = target[:,:,:,index]
    result_ = result[:,:,:,index]

    divisor = torch.sqrt(torch.tensor([n_lon*n_lat])).to(device)

    _, std = load_averages()
    std    = std.to(device)

    if not norm:
        std_multiplier = 1.0
    else:
        std_multiplier = std[index+norm_offset]
    # MSE = ((target - result) * stdev) ** 2
    
    mean_squared_error = ((target_ - result_) * std_multiplier / divisor)**2 # MSE has shape [n_patches, n_lat, n_lon]
    
    mean_squared_error = torch.einsum('i, aij->ai', weights, mean_squared_error).sum(dim=1)
    mean_squared_error = mean_squared_error.flatten().sum() 
    
    return mean_squared_error.item()
       
        
def calc_mse_single_variable_nonorm(target, result, device, weights, index=0):
    n_lon = 1440
    n_lat = len(weights)

    divisor = torch.sqrt(torch.tensor([n_lon*n_lat])).to(device)

    _, std = load_averages()
    std    = std.to(device)

    # MSE = ((target - result) * stdev) ** 2
    mean_squared_error = ((target[:, :, :, index] - result[:, :, :, index])/ divisor)**2

    mean_squared_error = torch.einsum('i, aij->ai', weights, mean_squared_error).sum(dim=1)
    mean_squared_error = mean_squared_error.flatten().sum() 
    
    return mean_squared_error.item()
    

def calc_acc_single_variable(config_dict, target, result, device, weights, index=0, norm=True):
    mean, std = load_averages()
    mean, std = mean.to(device), std.to(device)

    climatology = get_climatology()[:len(weights)].to(device)
    climatology = climatology - mean
    
    # If we do not need to normalize, the value of standard deviation is 1, does not affect results.
    if not norm:
        std[index] = 1

    if len(target.shape) == 4:
        target_anomoly = (target[:,:,:,index] * std[index].flatten() - climatology[:, :, index])
        result_anomoly = (result[:,:,:,index] * std[index].flatten() - climatology[:, :, index])
    else:
        target_anomoly = (target * std[index].flatten() - climatology[:, :, index])
        result_anomoly = (result * std[index].flatten() - climatology[:, :, index])
    numerator = torch.sum(torch.einsum('i,aij->ai', weights, torch.mul(target_anomoly, result_anomoly)), dim=(1))
    denom1    = torch.sum(torch.einsum('i,aij->ai', weights, target_anomoly**2), dim=(1))
    denom2    = torch.sum(torch.einsum('i,aij->ai', weights, result_anomoly**2), dim=(1))
    acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
    acc_sum   = torch.sum(acc_vec)
    
    return acc_sum.item()

def calc_acc_single_variable_parallel(config_dict, target, result, device, weights, index=0, norm=False, norm_offset=0, rank=0):
    if config_dict is None:
        n_lon = 1440
        n_lat = 721
    else:
        n_lon = config_dict['data']['xlon'] // 2
        n_lat = config_dict['data']['xlat']
    # per process: target has shape [n_patches, nlat, nlon//2, nvar/2]
    target_ = target[:,:,:,index]
    result_ = result[:,:,:,index]

    mean, std = load_averages()
    mean, std = mean.to(device), std.to(device)
    if rank % 4 == 0:
        climatology = get_climatology()[:n_lat, :n_lon, index+norm_offset].to(device)
    elif rank % 4 == 1:
        climatology = get_climatology()[:n_lat, :n_lon, index+norm_offset].to(device)
    elif rank % 4== 2:
        climatology = get_climatology()[:n_lat, n_lon:, index+norm_offset].to(device)
    elif rank % 4 == 3:
        climatology = get_climatology()[:n_lat, n_lon:, index+norm_offset].to(device)
        
    std_multiplier = std[index+norm_offset] if norm else 1.0
    
    target_anomoly = (target_ * std_multiplier - climatology)
    result_anomoly = (result_ * std_multiplier - climatology)
    
    numerator = torch.sum(torch.einsum('i,aij->ai', weights, torch.mul(target_anomoly, result_anomoly)), dim=(1))
    denom1    = torch.sum(torch.einsum('i,aij->ai', weights, target_anomoly**2), dim=(1))
    denom2    = torch.sum(torch.einsum('i,aij->ai', weights, result_anomoly**2), dim=(1))
    
    acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
    acc_sum   = torch.sum(acc_vec)
    return acc_sum.item()