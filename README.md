# TF2-Keras-LFADs
TF2+Keras LFADs for HLS4ML

This TF2+Keras LFADs aims to be pushed through HLS4ML flow and load onto a FPGA. The model and dataset are borrowed from HenningLab.The original repo (https://github.com/HennigLab/tndm) is aimed for showing the tndm (targeted Neural Dynamical Modeling) model. But the LFADs is also reimplemented in TF2+Keras in their work. For our research insterest, we'll only play with the TF2+Keras LFADs 

The TF2+Keras LFADs model is under tndm/models/lfads.py

# Download Miniconda & Create Environment 
Download and install minicoda if you have not

Then run

`conda create env -f environment.yaml`

Check and active envirmonment by

`conda env list`

`conda activate TF2Lfads`

Installing pandas 1.0.1 first

`pip install pandas==1.0.1`

Then run

`pip install -e .`

Finish environment setup

# Build & Run Model
Build the model berfore play with it

`python setup.py build`

Then, play with the TF2+Keras LFADs around by openning the notebooks

