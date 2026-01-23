#!/bin/bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
conda env create --name base --file=environment.yaml
pip install --no-build-isolation -e .
