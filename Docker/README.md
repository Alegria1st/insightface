# Dockers Version 1.1

nvcr.io/nvidia/pytorch:21.11-py3 # insightface arcface require pytorch>= 1.9 version
opencv  ==4.4.0.46 # fix specific opencv because opencv > 4.5 error while debugging
tensorboard, PrettyTable, menpo # add packages
