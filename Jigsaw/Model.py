import numpy as np
import torch

from . import Dataset, Linear, PatchEmbedding
from . import Normalization as Norm
from .Mixer import MixerBlock, MixerBlockMegatron, MixerBlockSequential


class WeatherMixer(torch.nn.Module):
    def __init__(
        self, config_dict, rank, device, parallelism, model_group, col_subgroup
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.column_subgroup = col_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]

        if (
            config_dict["model"]["patch_embed"] == 1
        ):  # 2-dimensional convolutional patch embedding
            self.patch_size = config_dict["model"]["patch_size"]
            self.stride = config_dict["model"]["stride"]
            self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
                self.xlat,
                self.xlon,
                self.patch_size,
                72,
                hidden_dimension,
                self.rank,
                self.device,
                parallelism,
                model_group,
            )
        else:  # 2-dimensional convolutional patch embedding
            self.patch_size = (
                config_dict["model"]["patch_size"],
                config_dict["model"]["patch_size"],
            )
            self.n_lat, self.n_lon = (
                self.xlat // self.patch_size[0],
                self.xlon // self.patch_size[1],
            )
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
                self.xlat,
                self.xlon,
                self.patch_size,
                72,
                hidden_dimension,
                self.rank,
                self.device,
                parallelism,
                model_group,
                bias=True,
            )

        self.n_blocks = config_dict["model"]["mixing_blocks"]

        self.blocks = torch.nn.ModuleList(
            [
                MixerBlock(
                    self.n_lat * self.n_lon,
                    hidden_dimension,
                    config_dict["model"]["spatial_hidden_dim_fraction"],
                    config_dict["model"]["features_hidden_dim_fraction"],
                    rank,
                    self.device,
                    parallelism,
                    model_group,
                )
                for i in range(self.n_blocks)
            ]
        )

        # Define layer norms
        match parallelism:
            case 2:
                self.layer_norm = Linear.torch.nn.LayerNorm(hidden_dimension // 2)
                self.num_variables = 72
            case 4 | 8:
                self.layer_norm = Norm.DistributedLayerNorm(
                    hidden_dimension // 2, self.rank, self.device, self.column_subgroup
                )
                self.num_variables = 36  # TODO: generalize via config file

        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]

        self.add = config_dict["model"]["add"]

        if config_dict["model"]["patch_embed"] == 1:
            self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                self.patch_size,
                hidden_dimension,
                self.num_variables,
                rank,
                self.device,
                parallelism,
                model_group,
                concat=self.concat,
                stride=self.stride,
                bias=False,
            )
        else:  # 2-dimensional convolutional patch recovery
            self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                self.patch_size,
                hidden_dimension,
                self.num_variables,
                rank,
                self.device,
                parallelism,
                model_group,
            )

        if config_dict["model"]["linear_residual"]:
            self.linearlayer = Linear.DistributedXWT(
                self.num_variables * 2 if parallelism > 2 else self.num_variables,
                self.num_variables // 2 if parallelism == 8 else self.num_variables,
                rank,
                self.device,
                parallelism,
                model_group,
            )
            if parallelism == 2:
                self.layer_norm2 = torch.nn.LayerNorm(self.num_variables // 2)
                self.layer_norm3 = torch.nn.LayerNorm(self.num_variables)
            else:
                self.layer_norm2 = Norm.DistributedLayerNorm(
                    self.num_variables, self.rank, self.device, self.column_subgroup
                )
                self.layer_norm3 = Norm.DistributedLayerNorm(
                    self.num_variables * 2, self.rank, self.device, self.column_subgroup
                )

        self.activation = torch.nn.GELU()
        self.model_group = model_group

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )

        # Estimate the number of FLOPs/forward pass and intra-node communication
        self._load_constant_masks(config_dict, parallelism)
        self._calc_flop_comm(config_dict)

    def _load_constant_masks(self, config_dict, parallelism):
        match parallelism:
            case 1 | 2:
                self.start, self.end = 0, self.xlon
            case 4:
                match self.rank:
                    case 0 | 1:
                        self.start, self.end = 0, self.xlon // 2
                    case 2 | 3:
                        self.start, self.end = self.xlon // 2, self.xlon
            case 8:
                match self.rank:
                    case 0 | 1:
                        self.start, self.end = 0, self.xlon // 4
                    case 2 | 3:
                        self.start, self.end = self.xlon // 4, self.xlon // 2
                    case 4 | 5:
                        self.start, self.end = self.xlon // 2, self.xlon * 3 // 4
                    case 6 | 7:
                        self.start, self.end = self.xlon * 3 // 4, self.xlon

        self.land_mask, self.soil_type, self.topography = Dataset.load_constant_mask_2d(
            config_dict["data"]["constant_masks_path"], xlat=self.xlat
        )
        self.land_mask = torch.unsqueeze(self.land_mask[:, self.start : self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:, self.start : self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:, self.start : self.end], 0)

    def _calc_flop_comm(self, config_dict):
        nlat, nlon = self.xlat, self.xlon
        parallelism = self.parallelism
        nvars_in, nvars_out = 72, 72
        patch_size = config_dict["model"]["patch_size"]
        patch_embedding_in = patch_size * nvars_in
        patch_embedding_out = config_dict["model"]["hidden_dim"]
        # where ntokens = nlat*nlon/patch_size
        ntokens = self.xlat * self.xlon / patch_size
        spatial_hidden_dim = int(
            config_dict["model"]["spatial_hidden_dim_fraction"] * ntokens
        )
        channels_hidden_dim = int(
            config_dict["model"]["features_hidden_dim_fraction"] * patch_embedding_out
        )

        # patch embedding
        patch_embedding_comm = flop_comm_estimation.AB_communication(
            parallelism, ntokens, patch_size * nvars_in, patch_embedding_out
        )
        patch_embedding_flop = (
            flop_comm_estimation.AB_flop_local(
                ntokens, patch_size * nvars_in, patch_embedding_out
            )
            + patch_embedding_out / 1e12
        )

        # self.concat + 1 (multiplier = 1 if no concatenation, = 2 if concat)
        patch_recovery_comm = flop_comm_estimation.AB_communication(
            parallelism,
            ntokens,
            patch_embedding_out * (self.concat + 1),
            patch_embedding_in,
        )
        patch_recovery_flop = (
            flop_comm_estimation.AB_flop_local(
                ntokens, patch_embedding_out * (self.concat + 1), patch_embedding_in
            )
            + patch_embedding_in / 1e12
        )

        linlayer_comm = flop_comm_estimation.AB_communication(
            parallelism, nlat * nlon, nvars_out * 2, nvars_out
        )
        linlayer_flop = (
            flop_comm_estimation.AB_flop_local(nlat * nlon, nvars_out * 2, nvars_out)
            + nvars_out / 1e12
        )

        tflop_per_block, gb_per_block = flop_comm_estimation.mixing_block(
            parallelism,
            ntokens,
            patch_embedding_out,
            spatial_hidden_dim,
            channels_hidden_dim,
        )

        flop_forward = (
            tflop_per_block * config_dict["model"]["mixing_blocks"]
            + patch_embedding_flop
            + linlayer_flop
            + patch_recovery_flop
        )
        comm_forward = (
            gb_per_block * config_dict["model"]["mixing_blocks"]
            + patch_embedding_comm
            + linlayer_comm
            + patch_recovery_comm
        )
        if self.rank == 0:
            print(f"Forward TFLOPS: {flop_forward} and comm (GB) {comm_forward}.")

        return flop_forward, comm_forward

    def forward(self, x, rollout_=0):
        print(f'hello')
        for i in range(rollout_):
            x_res = x
            
            if len(x_res.shape) > 3:
                x_res = x_res.flatten(start_dim=1, end_dim=2)
            else:
                x = x.reshape(x.shape[0], self.n_lat, self.n_lat, x.shape[-1])

                if self.rank % 2 == 1:
                    # change last 3 dims to constant mask
                    x[..., -3] = self.land_mask
                    x[..., -2] = self.soil_type
                    x[..., -1] = self.topography
                    x_res = x.flatten(start_dim=1, end_dim=2)
            # x_shape [n_batch, n_var, 720, 1440]
            x_res = x.flatten(start_dim=1, end_dim=2)
            x = self.embedding(x)
            x_res2 = x

            for i in range(self.n_blocks):
                x = self.blocks[i](x)

            # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2]
            # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
            if self.concat:
                x = self.layer_norm(x)
                x = torch.concat((x, x_res2), dim=2)
            else:
                x = self.layer_norm(x + x_res2)

            x = self.recovery(x, self.n_lat, self.n_lon // (self.parallelism // 2))
            x = x.reshape(x.shape[0], -1, x.shape[-1])

            if self.add:
                return x + x_res

            if self.linear_residual:
                x = self.layer_norm2(x)
                x = torch.concat((x, x_res), dim=2)

                # After recovery: [n_batch, n_lat, n_lon, n_var]
                x = self.layer_norm3(x)
                x = x.reshape(x.shape[0], -1, x.shape[-1])
                x = self.linearlayer(x)
            print(x)
        return x


class Model_eightway(torch.nn.Module):
    def __init__(
        self, config_dict, rank, device, parallelism, model_group, col_subgroup
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.column_subgroup = col_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]

        if config_dict["model"]["patch_embed"] == 1:
            self.patch_size = config_dict["model"]["patch_size"]
            self.stride = config_dict["model"]["stride"]
            self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
                self.xlat,
                self.xlon,
                self.patch_size,
                72,
                hidden_dimension,
                self.rank,
                self.device,
                parallelism,
                model_group,
            )
        else:
            self.patch_size = (
                config_dict["model"]["patch_size"],
                config_dict["model"]["patch_size"],
            )
            self.n_lat, self.n_lon = (
                self.xlat // self.patch_size[0],
                self.xlon // self.patch_size[1],
            )
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
                self.xlat,
                self.xlon,
                self.patch_size,
                72,
                hidden_dimension,
                self.rank,
                self.device,
                parallelism,
                model_group,
            )

        self.n_blocks = config_dict["model"]["mixing_blocks"]

        self.blocks = torch.nn.ModuleList(
            [
                MixerBlock(
                    self.n_lat * self.n_lon,
                    hidden_dimension,
                    config_dict["model"]["spatial_hidden_dim_fraction"],
                    config_dict["model"]["features_hidden_dim_fraction"],
                    rank,
                    self.device,
                    parallelism,
                    model_group,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = Norm.DistributedLayerNorm(
            hidden_dimension // 2, self.rank, self.device, self.column_subgroup
        )

        self.num_variables = 36  # TODO: generalize via config file

        self.layer_norm2 = Norm.DistributedLayerNorm(
            self.num_variables, self.rank, self.device, self.column_subgroup
        )
        self.layer_norm3 = Norm.DistributedLayerNorm(
            self.num_variables * 2, self.rank, self.device, self.column_subgroup
        )
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]

        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.add = config_dict["model"]["add"]
        if config_dict["model"]["patch_embed"] == 1:
            if self.concat:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                    self.patch_size,
                    hidden_dimension,
                    self.num_variables,
                    rank,
                    self.device,
                    parallelism,
                    model_group,
                    concat=True,
                    stride=self.stride,
                )
            else:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                    self.patch_size,
                    hidden_dimension,
                    self.num_variables,
                    rank,
                    self.device,
                    parallelism,
                    model_group,
                    concat=False,
                    stride=self.stride,
                )
        else:
            self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                self.patch_size,
                hidden_dimension,
                self.num_variables,
                rank,
                self.device,
                parallelism,
                model_group,
            )

        self.linearlayer = Linear.DistributedXWT(
            self.num_variables * 2,
            self.num_variables // 2,
            rank,
            self.device,
            parallelism,
            model_group,
        )

        self.activation = torch.nn.GELU()
        self.model_group = model_group

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )
        self.land_mask, self.soil_type, self.topography = Dataset.load_constant_mask_2d(
            config_dict["data"]["constant_masks_path"], xlat=self.xlat
        )
        if self.rank % 8 in [0, 1]:
            self.start, self.end = 0, self.xlon // 4
        elif self.rank % 8 in [2, 3]:
            self.start, self.end = self.xlon // 4, self.xlon // 2
        elif self.rank * 8 in [4, 5]:
            self.start, self.end = self.xlon // 2, self.xlon // 4 * 3
        else:
            self.start, self.end = self.xlon // 4 * 3, self.xlon
        self.land_mask = torch.unsqueeze(self.land_mask[:, self.start : self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:, self.start : self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:, self.start : self.end], 0)

    def forward(self, x, rollout_=0):
        for i in range(rollout_):
            x_res = x
            if len(x_res.shape) > 3:
                x_res = x_res.flatten(start_dim=1, end_dim=2)
            else:
                x = x.reshape(x.shape[0], self.n_lat, self.n_lat, x.shape[-1])

                if self.rank % 2 == 1:
                    # change last 3 dims to constant mask
                    x[..., -3] = self.land_mask
                    x[..., -2] = self.soil_type
                    x[..., -1] = self.topography
                    x_res = x.flatten(start_dim=1, end_dim=2)
            # x_shape [n_batch, n_var, 720, 1440]
            x = self.embedding(x)
            # After embedding: x_shape [n_batch, n_patch_lat * n_patch_lon, embed_dim]
            x_res2 = x

            for i in range(self.n_blocks):
                x = self.blocks[i](x)
                torch.cuda.synchronize()

            # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2]
            # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
            if self.concat:
                x = self.layer_norm(x)
                x = torch.concat((x, x_res2), dim=2)
            else:
                x = self.layer_norm(x + x_res2)
            x = self.recovery(x, self.n_lat, self.n_lon // 4)
            x = x.reshape(x.shape[0], -1, x.shape[-1])

            if self.add:
                return x + x_res

            if self.linear_residual:
                x = self.layer_norm2(x)
                x = torch.concat((x, x_res), dim=2)

                # After recovery: [n_batch, n_lat, n_lon, n_var]
                x = self.layer_norm3(x)
                x = x.reshape(x.shape[0], -1, x.shape[-1])
                x = self.linearlayer(x)
        return x


class Model(torch.nn.Module):
    def __init__(
        self, config_dict, rank, device, parallelism, model_group, col_subgroup
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.column_subgroup = col_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]

        if config_dict["model"]["patch_embed"] == 1:
            self.patch_size = config_dict["model"]["patch_size"]
            self.stride = config_dict["model"]["stride"]
            self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
                self.xlat,
                self.xlon,
                self.patch_size,
                72,
                hidden_dimension,
                self.rank,
                self.device,
                parallelism,
                model_group,
                bias=True,
            )
        else:
            self.patch_size = (
                config_dict["model"]["patch_size"],
                config_dict["model"]["patch_size"],
            )
            self.n_lat, self.n_lon = (
                self.xlat // self.patch_size[0],
                self.xlon // self.patch_size[1],
            )
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
                self.xlat,
                self.xlon,
                self.patch_size,
                72,
                hidden_dimension,
                self.rank,
                self.device,
                parallelism,
                model_group,
                bias=True,
            )

        self.n_blocks = config_dict["model"]["mixing_blocks"]

        self.blocks = torch.nn.ModuleList(
            [
                MixerBlock(
                    self.n_lat * self.n_lon,
                    hidden_dimension,
                    config_dict["model"]["spatial_hidden_dim_fraction"],
                    config_dict["model"]["features_hidden_dim_fraction"],
                    rank,
                    self.device,
                    parallelism,
                    model_group,
                )
                for i in range(self.n_blocks)
            ]
        )

        self.layer_norm = Norm.DistributedLayerNorm(
            hidden_dimension // 2, self.rank, self.device, self.column_subgroup
        )

        self.num_variables = 36  # TODO: generalize via config file

        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]

        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.add = config_dict["model"]["add"]
        if config_dict["model"]["patch_embed"] == 1:
            if self.concat:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                    self.patch_size,
                    hidden_dimension,
                    self.num_variables,
                    rank,
                    self.device,
                    parallelism,
                    model_group,
                    concat=True,
                    stride=self.stride,
                    bias=False,
                )
            else:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                    self.patch_size,
                    hidden_dimension,
                    self.num_variables,
                    rank,
                    self.device,
                    parallelism,
                    model_group,
                    concat=False,
                    stride=self.stride,
                )
        else:
            self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                self.patch_size,
                hidden_dimension,
                self.num_variables,
                rank,
                self.device,
                parallelism,
                model_group,
            )

        if config_dict["model"]["linear_residual"]:
            self.linearlayer = Linear.DistributedXWT(
                self.num_variables * 2,
                self.num_variables,
                rank,
                self.device,
                parallelism,
                model_group,
                bias=True,
            )
            self.layer_norm2 = Norm.DistributedLayerNorm(
                self.num_variables, self.rank, self.device, self.column_subgroup
            )
            self.layer_norm3 = Norm.DistributedLayerNorm(
                self.num_variables * 2, self.rank, self.device, self.column_subgroup
            )

        self.activation = torch.nn.GELU()
        self.model_group = model_group

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )
        self.land_mask, self.soil_type, self.topography = Dataset.load_constant_mask_2d(
            config_dict["data"]["constant_masks_path"], xlat=self.xlat
        )
        if self.rank % 4 in [0, 1]:
            self.start, self.end = 0, self.xlon // 2
        else:
            self.start, self.end = self.xlon // 2, self.xlon

        self.land_mask = torch.unsqueeze(self.land_mask[:, self.start : self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:, self.start : self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:, self.start : self.end], 0)
        self.flop_comm = self._calc_flop_comm(config_dict)

    def _calc_flop_comm(self, config_dict):
        nlat, nlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        parallelism = 4
        nvars_in, nvars_out = 72, 72
        patch_size = config_dict["model"]["patch_size"]
        patch_embedding_in = patch_size * nvars_in
        patch_embedding_out = config_dict["model"]["hidden_dim"]
        # where ntokens = nlat*nlon/patch_size
        input_data = (nlat * nlon / patch_size, patch_size * nvars_in)
        ntokens = nlat * nlon / patch_size
        spatial_hidden_dim = int(
            config_dict["model"]["spatial_hidden_dim_fraction"] * ntokens
        )
        channels_hidden_dim = int(
            config_dict["model"]["features_hidden_dim_fraction"] * patch_embedding_out
        )

        patch_embedding_comm = flop_comm_estimation.AB_communication(
            parallelism, ntokens, patch_size * nvars_in, patch_embedding_out
        )  # GBs of data communicated
        patch_embedding_flop = (
            flop_comm_estimation.AB_flop_local(
                parallelism, ntokens, patch_size * nvars_in, patch_embedding_out
            )
            + patch_embedding_out / 1e12
        )

        # including concat
        # (ntokens, patch_embedding_out)
        patch_recovery_comm = flop_comm_estimation.AB_communication(
            parallelism, ntokens, patch_embedding_out * 2, patch_embedding_in
        )  # GBs of data communicated
        patch_recovery_flop = (
            flop_comm_estimation.AB_flop_local(
                ntokens, patch_embedding_out * 2, patch_embedding_in
            )
            + patch_embedding_in / 1e12
        )

        linlayer_comm = flop_comm_estimation.AB_communication(
            parallelism, nlat * nlon, nvars_out * 2, nvars_out
        )  # GBs of data communicated
        linlayer_flop = (
            flop_comm_estimation.AB_flop_local(
                parallelism, nlat * nlon, nvars_out * 2, nvars_out
            )
            + nvars_out / 1e12
        )

        tflop_per_block, gb_per_block = flop_comm_estimation.mixing_block(
            parallelism,
            input_data[0],
            patch_embedding_out,
            spatial_hidden_dim,
            channels_hidden_dim,
        )

        flop_forward = (
            tflop_per_block * config_dict["model"]["mixing_blocks"]
            + patch_embedding_flop
            + linlayer_flop
            + patch_recovery_flop
        )
        comm_forward = (
            gb_per_block * config_dict["model"]["mixing_blocks"]
            + patch_embedding_comm
            + linlayer_comm
            + patch_recovery_comm
        )
        return flop_forward, comm_forward

    def forward(self, x, rollout_=0):
        x_res = x
        if len(x_res.shape) > 3:
            x_res = x_res.flatten(start_dim=1, end_dim=2)
        else:
            x = x.reshape(x.shape[0], self.n_lat, self.n_lat, x.shape[-1])

            if self.rank % 2 == 1:
                # change last 3 dims to constant mask
                x[..., -3] = self.land_mask
                x[..., -2] = self.soil_type
                x[..., -1] = self.topography
                x_res = x.flatten(start_dim=1, end_dim=2)

        # x_shape [n_batch, n_var, 720, 1440]
        x = self.embedding(x)
        # After embedding: x_shape [n_batch, n_patch_lat * n_patch_lon, embed_dim]
        x_res2 = x
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2]
        # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)

        x = self.recovery(x, self.n_lat, self.n_lon // 2)

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.add:
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)
            x = torch.concat((x, x_res), dim=2)

            # After recovery: [n_batch, n_lat, n_lon, n_var]
            x = self.layer_norm3(x)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = self.linearlayer(x)

        return x


class Model_twoway(torch.nn.Module):
    def __init__(self, config_dict, rank, device, parallelism, model_group):
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group  # Local groups [0,1], [2,3]...
        hidden_dimension = config_dict["model"]["hidden_dim"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        self.dropout = config_dict["model"]["dropout"]

        if config_dict["model"]["patch_embed"] == 1:
            self.patch_size = config_dict["model"]["patch_size"]
            self.stride = config_dict["model"]["stride"]
            self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
                self.xlat,
                self.xlon,
                self.patch_size,
                72,
                hidden_dimension,
                self.rank,
                self.device,
                parallelism,
                model_group,
                bias=True,
            )
        self.n_blocks = config_dict["model"]["mixing_blocks"]

        self.blocks = torch.nn.ModuleList(
            [
                MixerBlock(
                    self.n_lat * self.n_lon,
                    hidden_dimension,
                    config_dict["model"]["spatial_hidden_dim_fraction"],
                    config_dict["model"]["features_hidden_dim_fraction"],
                    rank,
                    device,
                    parallelism,
                    model_group,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = Linear.torch.nn.LayerNorm(hidden_dimension // 2)

        self.num_variables = 72  # TODO: generalize via config file

        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]

        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.add = config_dict["model"]["add"]
        if config_dict["model"]["patch_embed"] == 1:
            if self.concat:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                    self.patch_size,
                    hidden_dimension,
                    self.num_variables,
                    rank,
                    device,
                    parallelism,
                    model_group,
                    concat=True,
                    stride=self.stride,
                    bias=False,
                )
            else:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                    self.patch_size,
                    hidden_dimension,
                    self.num_variables,
                    rank,
                    device,
                    parallelism,
                    model_group,
                    concat=False,
                    stride=self.stride,
                    bias=False,
                )

        if config_dict["model"]["linear_residual"]:
            self.linearlayer = Linear.DistributedXWT(
                self.num_variables,
                self.num_variables,
                rank,
                device,
                parallelism=parallelism,
                model_group=model_group,
                bias=True,
            )
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables // 2)
            self.layer_norm3 = torch.nn.LayerNorm(self.num_variables)

        self.activation = torch.nn.GELU()

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )

    def forward(self, x, rollout_=0):
        x_res = x
        x_res = x_res.flatten(start_dim=1, end_dim=2)

        # x res for each GPU: [n_batch, n_lat, n_lon, n_var//4]
        # x_shape [n_batch, n_var, 720, 1440]
        x = self.embedding(x)
        # After embedding: x_shape [n_batch, n_patch_lat * n_patch_lon, embed_dim]
        x_res2 = x

        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2
        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)

        x = self.recovery(x, self.n_lat, self.n_lon)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.add:
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)
            x = torch.concat((x, x_res), dim=2)

            # After recovery: [n_batch, n_lat, n_lon, n_var]
            x = self.layer_norm3(x)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = self.linearlayer(x)
        return x


class ModelSequential(torch.nn.Module):
    def __init__(self, config_dict, rank, device):
        super().__init__()
        self.rank = rank
        hidden_dimension = config_dict["model"]["hidden_dim"]

        self.stride = config_dict["model"]["stride"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        padding = (0, 0)

        self.patch_embed = config_dict["model"]["patch_embed"]
        if self.patch_embed == 1:
            self.patch_size = config_dict["model"]["patch_size"]
            self.stride = config_dict["model"]["stride"]
            self.n_patches = (
                self.xlat * self.xlon - 1 * (self.patch_size - 1) - 1
            ) // self.stride + 1

            self.dropout = config_dict["model"]["dropout"]
            self.embedding = PatchEmbedding.PatchEmbedding1D(
                config_dict,
                self.patch_size,
                hidden_dimension,
                device=device,
                in_channels=72,
                padding=padding,
                stride=self.stride,
                xlat=self.xlat,
            )
            self.n_blocks = config_dict["model"]["mixing_blocks"]
            self.blocks = torch.nn.ModuleList(
                [
                    MixerBlockSequential(
                        self.n_patches,
                        hidden_dimension,
                        config_dict["model"]["spatial_hidden_dim_fraction"],
                        config_dict["model"]["features_hidden_dim_fraction"],
                        self.dropout,
                    )
                    for i in range(self.n_blocks)
                ]
            )
        elif self.patch_embed == 2:
            self.patch_size = (
                config_dict["model"]["patch_size"],
                config_dict["model"]["patch_size"],
            )
            self.embedding = PatchEmbedding.PatchEmbedding2D(
                self.patch_size,
                hidden_dimension,
                device=device,
                in_channels=72,
                padding=padding,
                stride=self.stride,
                xlat=self.xlat,
            )
            self.n_blocks = config_dict["model"]["mixing_blocks"]
            self.n_lat = (
                self.xlat + 2 * padding[0] - 1 * (self.patch_size[0] - 1) - 1
            ) // self.stride + 1
            self.n_lon = (
                self.xlon + 2 * padding[1] - 1 * (self.patch_size[1] - 1) - 1
            ) // self.stride + 1
            self.blocks = torch.nn.ModuleList(
                [
                    MixerBlockSequential(
                        self.n_lat * self.n_lon,
                        hidden_dimension,
                        hidden_dimension,
                        self.dropout,
                    )
                    for i in range(self.n_blocks)
                ]
            )
        else:
            self.patch_size = config_dict["model"]["patch_size"]
            # (self.xlat+2*padding[0]-1*(self.patch_size[0]-1)-1)//self.stride+1
            self.n_lat = 720 // 1
            # (self.xlon+2*padding[1]-1*(self.patch_size[1]-1)-1)//self.stride+1
            self.n_lon = 1440 // self.patch_size
            self.dropout = config_dict["model"]["dropout"]
            self.embedding = PatchEmbedding.PatchEmbeddingTest(
                self.patch_size,
                hidden_dimension,
                device=device,
                in_channels=72,
                padding=padding,
                stride=self.stride,
                xlat=self.xlat,
            )
            self.n_blocks = config_dict["model"]["mixing_blocks"]
            self.blocks = torch.nn.ModuleList(
                [
                    MixerBlockSequential(
                        720 * 1440 // self.patch_size,
                        hidden_dimension,
                        hidden_dimension,
                        self.dropout,
                    )
                    for i in range(self.n_blocks)
                ]
            )
        # self.pos_encoding = config_dict['model']['positional_encoding']
        #
        # if self.pos_encoding:
        #    self.positional_encoding = utils.getPositionEncoding(self.n_lat*self.n_lon, hidden_dimension, n=10000).to(device)

        self.latent_rollout = config_dict["model"]["latent_rollout"]

        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        if self.rank == 3:
            self.num_variables = 69
        else:
            self.num_variables = 69

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]
        if self.linear_residual:
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)
            self.layer_norm3 = torch.nn.LayerNorm(69 * 2)
            self.linearlayer = torch.nn.Linear(
                self.num_variables * 2, self.num_variables
            )

        if self.patch_embed == 1:
            if self.concat:
                self.recovery = PatchEmbedding.PatchRecovery1D(
                    self.patch_size,
                    hidden_dimension,
                    padding=padding,
                    stride=self.stride,
                    out_channels=69,
                    concat=True,
                    xlat=self.xlat,
                    xlon=self.xlon,
                )
            else:
                self.recovery = PatchEmbedding.PatchRecovery1D(
                    self.patch_size,
                    hidden_dimension,
                    padding=padding,
                    stride=self.stride,
                    out_channels=69,
                    concat=False,
                    xlat=self.xlat,
                    xlon=self.xlon,
                )
        elif self.patch_embed == 2:
            if self.concat:
                self.recovery = PatchEmbedding.PatchRecovery2D(
                    self.patch_size,
                    hidden_dimension,
                    padding=padding,
                    stride=self.stride,
                    out_channels=69,
                    concat=True,
                )
            else:
                self.recovery = PatchEmbedding.PatchRecovery2D(
                    self.patch_size,
                    hidden_dimension,
                    padding=padding,
                    stride=self.stride,
                    out_channels=69,
                    concat=False,
                )

        self.activation = torch.nn.GELU()

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )

    def forward(self, x, rollout_=1):
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)

        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)  # x: latent_t

        # x: shape [1, n_patches, hidden_dim]
        # if self.pos_encoding:
        #    x = x + self.positional_encoding

        x_res2 = x.clone()

        # x shape: [1, 1152, 2000]
        for rollout in torch.arange(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x)  # x: latent_tplus1

        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)

        if self.patch_embed == 2:  # Case: Decoder is 2D embedding
            x = self.recovery(
                x,
                self.n_lat,
                self.n_lon,
                output_size=(
                    x_res.shape[0],
                    x_res.shape[3],
                    x_res.shape[1],
                    x_res.shape[2],
                ),
            )
        else:  # Case: Decoder is 1D convolution
            x = self.recovery(x)

        if self.add:  # Recovery
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)  # layer norm must go before concat
            x = torch.concat((x[:, :, :, : self.num_variables], x_res[..., :69]), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat * self.xlon, self.num_variables * 2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
        return x


class ModelMegatron(torch.nn.Module):
    def __init__(self, config_dict, rank, device, parallelism, model_group):
        super().__init__()
        self.rank = rank
        hidden_dimension = config_dict["model"]["hidden_dim"]

        self.stride = config_dict["model"]["stride"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        padding = (0, 0)

        self.patch_embed = config_dict["model"]["patch_embed"]
        self.num_variables = 69

        self.patch_size = config_dict["model"]["patch_size"]
        self.stride = config_dict["model"]["stride"]
        self.n_patches = (
            self.xlat * self.xlon - 1 * (self.patch_size - 1) - 1
        ) // self.stride + 1
        #
        self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size

        self.dropout = config_dict["model"]["dropout"]
        self.embedding = PatchEmbedding.PatchEmbedding1D(
            config_dict,
            self.patch_size,
            hidden_dimension,
            device=device,
            in_channels=72,
            padding=padding,
            stride=self.stride,
            xlat=self.xlat,
            bias=False,
        )
        self.n_blocks = config_dict["model"]["mixing_blocks"]
        self.blocks = torch.nn.ModuleList(
            [
                MixerBlockMegatron(
                    self.n_lat * self.n_lon,
                    hidden_dimension,
                    config_dict["model"]["spatial_hidden_dim_fraction"],
                    config_dict["model"]["features_hidden_dim_fraction"],
                    rank,
                    device,
                    parallelism,
                    model_group,
                    self.dropout,
                    bias=False,
                )
                for i in range(self.n_blocks)
            ]
        )

        self.latent_rollout = config_dict["model"]["latent_rollout"]

        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]

        if self.linear_residual:
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)
            self.layer_norm3 = torch.nn.LayerNorm(self.num_variables * 2)
            self.linearlayer = torch.nn.Linear(
                self.num_variables * 2, self.num_variables, bias=False
            )

        if self.concat:
            self.recovery = PatchEmbedding.PatchRecovery1D(
                self.patch_size,
                hidden_dimension,
                padding=padding,
                stride=self.stride,
                out_channels=69,
                concat=True,
                xlat=self.xlat,
                xlon=self.xlon,
                bias=False,
            )
        else:
            self.recovery = PatchEmbedding.PatchRecovery1D(
                self.patch_size,
                hidden_dimension,
                padding=padding,
                stride=self.stride,
                out_channels=69,
                concat=False,
                xlat=self.xlat,
                xlon=self.xlon,
                bias=False,
            )

        self.activation = torch.nn.GELU()

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )

    def forward(self, x, rollout_=1):
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)

        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)  # x: latent_t

        x_res2 = x.clone()

        # x shape: [1, 1152, 2000]
        for rollout in torch.arange(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x)  # x: latent_tplus1

        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)

        x = self.recovery(x)

        if self.add:  # Recovery
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
            return x + x_res[..., :-3]

        if self.linear_residual:
            x = self.layer_norm2(x)  # layer norm must go before concat
            x = torch.concat(
                (x[:, :, :, : self.num_variables], x_res[..., : self.num_variables]),
                dim=3,
            )
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat * self.xlon, self.num_variables * 2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
        return x
