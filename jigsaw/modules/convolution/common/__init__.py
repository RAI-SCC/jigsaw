"""Pieces shared by the distributed convolution and transpose-convolution modules.

``arg_norm`` -- normalising the module's ``int`` / tuple / process-group
constructor arguments;
``spatial_split`` -- walking / configuring an ``n_spatial``-tuple domain split;
``channel_comm`` -- channel-parallel primitives (strategy resolution, per-hop
transport, full-channel gather, weight-block slicing);
``halo`` -- the uneven, Gloo-safe last-axis all-to-all both halo exchanges ride on.
"""
