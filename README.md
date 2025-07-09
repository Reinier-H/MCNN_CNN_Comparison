# Code created for the purpose of comparing the CNN2D and the MorphConvHypernet

The CNN2D code comes from https://github.com/mhaut/hyperspectral_deeplearning_review (GPL-3.0 licensed), from the following paper:

M. E. Paoletti, J. M. Haut, J. Plaza and A. Plaza.
Deep Learning Classifiers for Hyperspectral Imaging: A Review
International Society for Photogrammetry and Remote Sensing
DOI: 10.1016/j.isprsjprs.2019.09.006
vol. 158, pp. 279-317, December 2019.

The MorphConvHyperNet code was found at https://github.com/mhaut/hyperspectral_deeplearning_review (MIT Licensed), as the inofficial code for:

S. K. Roy, R. Mondal, M. E. Paoletti, J. M. Haut and A. Plaza
Morphological Convolutional Neural Networks for Hyperspectral Image Classification
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
DOI: 10.1109/JSTARS.2021.3088228
vol. 14, pp. 8689-8702, 2021.


The both_models folder contains the code, combined from both githubs, and adjusted for our purposes

inputData contains the Indian Pines (IP) and University of Pavia (UP) sets, along with their ground truth

Data_to_use is the folder where we have chosen to save data to use in our thesis, where files within subfolders are compared with eachother, using the results_to_csv.py

When running the code in both.py, where the main loop is contained, it forms folders in the main workspace, in the shape of "resultsyear-month-day" within which the results of that day are placed

The saved_models folder is filled, after using the retrain argument on both.py, storing MCNN and or CNN models, that have been trained on the IP set, to be tested with the UP set.

To run the models provided in this codebase, we recommend the nvcr.io/nvidia/tensorflow:25.02-tf2-py3 docker image, found at https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-02.html paired with docker desktop, when used on a Windows PC. 

After importing the image, form a container using the following command in powershell:
docker run -it --gpus all --name `<add new container name here>` -v `<add path to this folder here>`:/workspace:rw nvcr.io/nvidia/tensorflow:25.02-tf2-py3 bash

This commands does the following:
* -it: makes it interactive, allowing one to interact with it  in the terminal
* --gpus all: allows the docker container to make use of the gpu
* --name: allows one to reopen the container later
* --v: mounts the folder on the host pc onto the docker container, allowing one to edit on host, and run in container. places it in /workspace, with read and write permissions
* nvcr.io/nvidia/tensorflow:25.02-tf2-py3 is the image used to create the docker container
* bash tells it to open a bash shell for you to interact with

Afterwards, when in the workspace folder, you can use:
python3 both_models/both.py --dataset IP --repeat 1 --tr_percent 0.01 --use_val --set_parameters --batch_size 64 --epochs 1 --models MCNN CNN 2> error.txt

As an example on how to run the code training on 1% of the IP set, for 1 epoch, with 1 run, for both models. 
The 2> error.txt prevents the terminal from being flooded by warnings, and allows for finding out the cause of a crash, if the terminal has closed for some reason.
