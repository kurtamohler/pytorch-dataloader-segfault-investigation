This repo was created to help me attempt to reproduce an issue in PyTorch:
[link to issue](https://github.com/pytorch/pytorch/issues/31758)

## Prerequisite: Conda

Install Miniconda3: [link](https://docs.conda.io/en/latest/miniconda.html)

## Prerequisite: CUDA toolkits

The different environments in this repo require different CUDA versions. I used
these instructions to install multiple versions CUDA versions on my system:
[link](https://github.com/Quansight/dev-notes/blob/master/CUDA-installation-in-qgpu.md#installing-cuda-toolkits)

When creating an environment, if the required CUDA version is not found, an
error is thrown.

## Creating environments

Each of the environments in this repo have their own directory. They all
contain a `create-conda-env` script which creates the conda environment, sets
up variables to point to the proper CUDA version, and installs the corresponding
PyTorch version.

To create, for instance, env0, run the following:

```bash
$ ./env0/create-conda-env
```

This will create the conda environment `pytorch-dataloader-env0`. To activate it,
run:

```bash
$ conda activate pytorch-dataloader-env0
```

## Running the DataLoader segfault test

To run the DataLoader test, make sure one of the environments has been activated
and then run:

```bash
$ python test.py
```
