# TF2-Keras-LFADs
TF2+Keras LFADs for HLS4ML
This TF2+Keras LFADs aims to be pushed through HLS4ML flow and load onto a FPGA. The model is borrowed from https://github.com/HennigLab/tndm. 
The LFADs model is under tndm/model/lfads.py

# Download Miniconda & Create Environment 
First, download and install minicoda if you have not
Then first run
`conda create env -f environment.yaml`

Check and Active envirmonment by

`conda env list`

`conda activate TF2Lfads`

Installing pandas 1.0.1 first
`pip install pandas==1.0.1`

Then run
`pip intall -e .`

Finish environment setup

# Run Model
Play the TF2+Keras LFADs around by openning the notebooks

