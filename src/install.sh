#!/bin/bash

sudo apt-get update

sudo apt-get install python3.7

sudo apt-get install python3.7-venv

python3.7 -m venv ../venv

source ../venv/bin/activate

pip install numpy
pip install torch torchvision
pip install argparse
pip install csv