#!/bin/bash

echo "This script installs OASIS and adds to the pythonpath"

git clone https://github.com/j-friedrich/OASIS
cd OASIS
python setup.py build_ext --inplace
python setup.py clean --all

export PYTHONPATH=$(pwd):$PYTHONPATH
 