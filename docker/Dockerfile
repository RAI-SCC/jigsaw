FROM nvcr.io/nvidia/pytorch:25.08-py3

# install requirements
# hdf5
RUN pip install h5netcdf h5py

# huggingface
RUN pip install huggingface-hub

# pyaml
RUN pip install PyYAML

# timm
RUN pip install timm

# google cloud
RUN pip install gcsfs

# xarray
RUN pip install xarray

# zarr
RUN pip install zarr

# mlperf binding stuff
RUN pip install --ignore-installed "git+https://github.com/NVIDIA/mlperf-common.git"

# copy stuff
COPY LICENSE /opt/Jigsaw/LICENSE
COPY README.md /opt/Jigsaw/README.md
COPY pyproject.toml /opt/Jigsaw/pyproject.toml
COPY cases /opt/Jigsaw/cases
COPY experiments /opt/Jigsaw/experiments
COPY jigsaw /opt/Jigsaw/jigsaw
COPY utils /opt/Jigsaw/utils
RUN pip install --no-build-isolation /opt/Jigsaw

