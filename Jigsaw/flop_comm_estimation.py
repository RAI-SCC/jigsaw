# For an AB multiplication,
# define based on global in/hid/out dimensions


def AB_flop_global(in_dim, hidden_dim, out_dim):
    return in_dim * out_dim * (2 * hidden_dim - 1)


def AB_flop_local(in_dim, hidden_dim, out_dim):
    in_local, hidden_local, out_local = in_dim / 2, hidden_dim / 2, out_dim / 2
    flop = 4 * (
        2 * AB_flop_global(in_local, hidden_local, out_local) + in_local * out_local
    )
    return flop / 1e12


def AB_communication(parallelism, in_dim, hidden_dim, out_dim, dtype="float32"):
    in_local, hidden_local, out_local = in_dim / 2, hidden_dim / 2, out_dim / 2

    match parallelism:
        case 1:
            return 0
        case 2:
            return 2 * in_dim * hidden_local
        case 4:
            return 4 * (in_local * hidden_local + hidden_local * out_local)
        case 8:
            return in_dim * hidden_dim + 3 * hidden_dim * out_dim
    match dtype:
        case "float32":
            return comm / (2**30)
        case "float64":
            return comm / (2**30) * 2


def transpose_communication(in_dim, out_dim):
    in_local, out_local = in_dim / 2, out_dim / 2
    return in_local * out_local * 2 / (2**30)


def mixing_block(parallelism, ntokens, patch_embedding_out, spatial_hidden_dim, channels_hidden_dim):
    # input shape = output of patch embedding shape
    input_shape = (ntokens, patch_embedding_out)
    flop = 0
    comm = 0
    # Transpose
    comm = comm + transpose_communication(*input_shape)
    # Token mixing
    flop = (
        flop
        + AB_flop_local(patch_embedding_out, ntokens, spatial_hidden_dim)
        + spatial_hidden_dim / 1e12
    )
    comm = comm + AB_communication(parallelism, patch_embedding_out, ntokens, spatial_hidden_dim)
    # activation
    flop = flop + patch_embedding_out * spatial_hidden_dim / 1e12
    # token mixing 2
    flop = (
        flop
        + AB_flop_local(patch_embedding_out, spatial_hidden_dim, ntokens)
        + ntokens / 1e12
    )
    comm = comm + AB_communication(parallelism, patch_embedding_out, spatial_hidden_dim, ntokens)

    # transpose
    comm = comm + transpose_communication(patch_embedding_out, ntokens)

    # Residual connection addition
    flop = flop + ntokens * patch_embedding_out / 1e12

    # channel mixing
    flop = (
        flop
        + AB_flop_local(ntokens, patch_embedding_out, channels_hidden_dim)
        + channels_hidden_dim / 1e12
    )
    comm = comm + AB_communication(parallelism, ntokens, patch_embedding_out, channels_hidden_dim)
    # activation
    flop = flop + ntokens * channels_hidden_dim / 1e12
    # channel mixing 2
    flop = (
        flop
        + AB_flop_local(ntokens, channels_hidden_dim, patch_embedding_out)
        + patch_embedding_out / 1e12
    )
    comm = comm + AB_communication(parallelism, ntokens, channels_hidden_dim, patch_embedding_out)

    # residual addition
    flop = flop + ntokens * patch_embedding_out / 1e12

    return flop, comm  # FLOP: GigaFLOP, Comm: GBs


def transformer_block(parallelism, ntokens, hidden_dim, transformer_dim, out_dim, heads):
    # input shape = output of patch embedding shape
    input_shape = (ntokens, hidden_dim)
    flop = 0
    comm = 0

    # qkv calculation
    # X: (n_tokens, hidden_dim) @ (hidden_dim, 3 * transformer_dimension)
    flop = flop + AB_flop_local(ntokens, hidden_dim, 3 * transformer_dim)
    comm = comm + AB_communication(parallelism, ntokens, hidden_dim, 3 * transformer_dim)

    # Q @ K.mT
    # Q: (n_tokens, transformer_dim) K.mT : (transformer_dim, n_tokens)
    flop = flop + heads * AB_flop_local(ntokens, transformer_dim // heads, ntokens)
    # TODO: currently ignore softmax
    # (Q@K.mT) @ V
    # (Q@K.mT): (n_tokens, n_tokens) @ (n_tokens, transformer_dim)
    flop = flop + heads * AB_flop_local(ntokens, ntokens, transformer_dim // heads)

    # concatenator
    flop = flop + AB_flop_local(ntokens, transformer_dim, out_dim)

    return flop, comm  # FLOP: GigaFLOP, Comm: GBs


def calc_flops_comm(config):
    nlat, nlon = config["data"]["xlat"], config["data"]["xlon"]
    parallelism = config["modelparallel"]["parallelism"]
    nvars_in, nvars_out = 72, 72
    patch_size = config["model"]["patch_size"]
    patch_embedding_in = patch_size * nvars_in
    patch_embedding_out = config["model"]["hidden_dim"]
    # where ntokens = nlat*nlon/patch_size
    input_data = (nlat * nlon / patch_size, patch_size * nvars_in)
    ntokens = nlat * nlon / patch_size
    spatial_hidden_dim = int(config["model"]["spatial_hidden_dim_fraction"] * ntokens)
    channels_hidden_dim = int(
        config["model"]["features_hidden_dim_fraction"] * patch_embedding_out
    )

    patch_embedding_comm = AB_communication(
        parallelism, ntokens, patch_size * nvars_in, patch_embedding_out
    )  # GBs of data communicated
    patch_embedding_flop = (
        AB_flop_local(ntokens, patch_size * nvars_in, patch_embedding_out)
        + patch_embedding_out / 1e12
    )

    # including concat
    # (ntokens, patch_embedding_out)
    patch_recovery_comm = AB_communication(
        parallelism, ntokens, patch_embedding_out * 2, patch_embedding_in
    )  # GBs of data communicated
    patch_recovery_flop = (
        AB_flop_local(ntokens, patch_embedding_out * 2, patch_embedding_in)
        + patch_embedding_in / 1e12
    )

    linlayer_comm = AB_communication(
        nlat * nlon, nvars_out * 2, nvars_out
    )  # GBs of data communicated
    linlayer_flop = (
        AB_flop_local(nlat * nlon, nvars_out * 2, nvars_out) + nvars_out / 1e12
    )

    tflop_per_block, gb_per_block = mixing_block(
        input_data[0], patch_embedding_out, spatial_hidden_dim, channels_hidden_dim
    )

    flop_forward = (
        tflop_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_flop
        + linlayer_flop
        + patch_recovery_flop
    )
    comm_forward = (
        gb_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_comm
        + linlayer_comm
        + patch_recovery_comm
    )

    return flop_forward, comm_forward


def calc_flops_transformer(config):
    nlat, nlon = config["data"]["xlat"], config["data"]["xlon"]
    nvars_in, nvars_out = 72, 72
    patch_size = config["model"]["patch_size"]
    patch_embedding_in = patch_size * nvars_in
    patch_embedding_out = config["model"]["hidden_dim"]
    transformer_dim = config["model"]["transformer_dim"]
    # where ntokens = nlat*nlon/patch_size
    input_data = (nlat * nlon / patch_size, patch_size * nvars_in)
    ntokens = nlat * nlon / patch_size
    patch_embedding_comm = AB_communication(
        ntokens, patch_size * nvars_in, patch_embedding_out
    )  # GBs of data communicated
    patch_embedding_flop = (
        AB_flop_local(ntokens, patch_size * nvars_in, patch_embedding_out)
        + patch_embedding_out / 1e12
    )

    # including concat
    # (ntokens, patch_embedding_out)
    patch_recovery_comm = AB_communication(
        ntokens, patch_embedding_out * 2, patch_embedding_in
    )  # GBs of data communicated
    patch_recovery_flop = (
        AB_flop_local(ntokens, patch_embedding_out * 2, patch_embedding_in)
        + patch_embedding_in / 1e12
    )

    linlayer_comm = AB_communication(
        nlat * nlon, nvars_out * 2, nvars_out
    )  # GBs of data communicated
    linlayer_flop = (
        AB_flop_local(nlat * nlon, nvars_out * 2, nvars_out) + nvars_out / 1e12
    )

    tflop_per_block, gb_per_block = transformer_block(
        input_data[0],
        patch_embedding_out,
        transformer_dim,
        patch_embedding_out,
        config["model"]["heads"],
    )

    flop_forward = (
        tflop_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_flop
        + linlayer_flop
        + patch_recovery_flop
    )
    comm_forward = (
        gb_per_block * config["model"]["mixing_blocks"]
        + patch_embedding_comm
        + linlayer_comm
        + patch_recovery_comm
    )

    return flop_forward, comm_forward
