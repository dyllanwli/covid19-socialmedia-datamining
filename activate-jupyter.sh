#!/bin/bash

# This scripts is a happy path for running python program on HPRC using nohup

# module load Python/3.7.0-intel-2018b

# virtualenv ../HPRC-Python3.7-Example/venv

# source ../HPRC-Python3.7-Example/venv/bin/activate

module load Anaconda/3-5.0.0.1
source activate jupyterlab_1.2.2


# nohup python test-tweets.py & >> nohup.out

######### Deactivate
# deactivate 
