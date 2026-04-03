#!/bin/bash
# Build the PyTorch CUDA extension.
# Run from a GPU node that has CUDA and the project120 venv active.
#
# Usage:
#   source /pub/<yourUser>/project120/bin/activate
#   bash build.sh

TORCH_CUDA_ARCH_LIST="8.0 7.0" python setup.py install --verbose
