import numpy as np
import torch

import Linear
import PatchEmbedding
import Attention
import MLP
import Dataset
import Normalization as Norm


class Transformer_Sequential(torch.nn.Module):
    def __init__(
        self,
        config_dict,
        rank,
        device,
        n_parallel,
        model_group,
        row_subgroup,
        col_subgroup,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.column_subgroup = col_subgroup
        self.softmax_subgroup = row_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        transformer_dimension = config_dict["model"]["transformer_dim"]
        heads = config_dict["model"]["heads"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        padding = (0, 0)
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
                Attention.DistributedTransformer(
                    hidden_dimension,
                    transformer_dimension,
                    hidden_dimension,
                    heads,
                    rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    self.softmax_subgroup,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
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
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)  # layer norm must go before concat
            x = torch.concat((x[:, :, :, : self.num_variables], x_res[..., :69]), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat * self.xlon, self.num_variables * 2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
        return x


class Transformer_Twoway(torch.nn.Module):
    def __init__(
        self,
        config_dict,
        rank,
        device,
        n_parallel,
        model_group,
        row_subgroup,
        col_subgroup,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.column_subgroup = col_subgroup
        self.softmax_subgroup = row_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        transformer_dimension = config_dict["model"]["transformer_dim"]
        heads = config_dict["model"]["heads"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        self.patch_size = config_dict["model"]["patch_size"]
        self.stride = config_dict["model"]["stride"]
        self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
        self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
            self.xlat,
            self.xlon,
            self.patch_size,
            72,
            hidden_dimension,
            rank,
            device,
            n_parallel,
            model_group,
            bias=False
        )
        self.n_blocks = config_dict["model"]["mixing_blocks"]
        self.blocks = torch.nn.ModuleList(
            [
                Attention.DistributedTransformer(
                    hidden_dimension,
                    transformer_dimension,
                    hidden_dimension,
                    heads,
                    rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    self.softmax_subgroup,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension // 2)
        self.num_variables = 72

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]
        if self.linear_residual:
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables // 2)
            self.layer_norm3 = torch.nn.LayerNorm(self.num_variables)
            self.linearlayer = Linear.DistributedXWT(
                self.num_variables,
                self.num_variables,
                rank,
                device,
                n_parallel=n_parallel,
                model_group=model_group,
                bias=False
            )

        if self.concat:
            self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                self.patch_size,
                hidden_dimension,
                self.num_variables,
                rank,
                device,
                n_parallel,
                model_group,
                concat=True,
                stride=self.stride,
                bias=False
            )
        else:
            self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                self.patch_size,
                hidden_dimension,
                self.num_variables,
                rank,
                device,
                n_parallel,
                model_group,
                concat=False,
                stride=self.stride,
                bias=False
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
        x_res = x
        x_res = x_res.flatten(start_dim=1, end_dim=2)

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
            # x = x.flatten(start_dim=1, end_dim=2)
            x = self.linearlayer(x)

            return x
        else:
            return x


class Transformer_Fourway(torch.nn.Module):
    def __init__(
        self,
        config_dict,
        rank,
        device,
        n_parallel,
        model_group,
        row_subgroup,
        col_subgroup,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.column_subgroup = col_subgroup
        self.softmax_subgroup = row_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        transformer_dimension = config_dict["model"]["transformer_dim"]
        heads = config_dict["model"]["heads"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        self.patch_size = config_dict["model"]["patch_size"]
        self.stride = config_dict["model"]["stride"]
        self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
        self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(
            self.xlat,
            self.xlon,
            self.patch_size,
            72,
            hidden_dimension,
            rank,
            device,
            n_parallel,
            model_group,
        )
        self.n_blocks = config_dict["model"]["mixing_blocks"]
        self.blocks = torch.nn.ModuleList(
            [
                Attention.TransformerLayer( # Transformer layer including MLP block
                    hidden_dimension,
                    transformer_dimension,
                    hidden_dimension,
                    heads,
                    rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    self.softmax_subgroup,
                    self.column_subgroup
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = Norm.DistributedLayerNorm(
            hidden_dimension // 2, self.rank, self.device, self.column_subgroup
        )
        self.num_variables = 36  # TODO: generalize via config file

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]
        if self.linear_residual:
            self.layer_norm2 = Norm.DistributedLayerNorm(
                self.num_variables, self.rank, self.device, self.column_subgroup
            )
            self.layer_norm3 = Norm.DistributedLayerNorm(
                self.num_variables * 2, self.rank, self.device, self.column_subgroup
            )
            self.linearlayer = Linear.DistributedXWT(
                self.num_variables * 2,
                self.num_variables,
                rank,
                self.device,
                n_parallel,
                model_group,
            )

        if self.concat:
            self.recovery = PatchEmbedding.Distributed1DPatchRecovery(
                self.patch_size,
                hidden_dimension,
                self.num_variables,
                rank,
                self.device,
                n_parallel,
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
                n_parallel,
                model_group,
                concat=False,
                stride=self.stride,
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

    def forward(self, x, rollout_=1):
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

        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)  # x: latent_t

        x_res2 = x.clone()

        # x shape: [1, 1152, 2000]
        for i in range(self.n_blocks):
            x = self.blocks[i](x)  # x: latent_tplus1

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
        else:
            return x


## Megatron Implementations
class Megatron_Transformer(torch.nn.Module):
    def __init__(
        self,
        config_dict,
        rank,
        device,
        n_parallel,
        model_group,
        row_subgroup,
        col_subgroup,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.column_subgroup = col_subgroup
        self.softmax_subgroup = row_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        transformer_dimension = config_dict["model"]["transformer_dim"]
        heads = config_dict["model"]["heads"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        self.patch_size = config_dict["model"]["patch_size"]
        self.stride = config_dict["model"]["stride"]
        self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
        padding = (0, 0)
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
                Attention.MegatronTransformerLayer(
                    hidden_dimension,
                    transformer_dimension,
                    hidden_dimension,
                    heads,
                    rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    self.softmax_subgroup,
                    self.column_subgroup
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        self.num_variables = 69

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]
        if self.linear_residual:
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)
            self.layer_norm3 = torch.nn.LayerNorm(self.num_variables * 2)
            self.linearlayer = torch.nn.Linear(
                self.num_variables * 2,
                self.num_variables,
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
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)  # layer norm must go before concat
            x = torch.concat((x[:, :, :, : self.num_variables], x_res[..., :69]), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat * self.xlon, self.num_variables * 2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)

        return x
