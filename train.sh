#!/bin/bash
sudo docker run -it --runtime=nvidia --privileged --shm-size=30g -v "$PWD":"$PWD" \
-v "/home/ddangelo/Documents/Tensorflow/rbc-ckpts":"/home/ddangelo/Documents/Tensorflow/rbc-ckpts" \
-w "$PWD" \
rbc \
mpiexec -n 4 --oversubscribe --allow-run-as-root \
python3 -u "/home/ddangelo/Dropbox/Deep Learning/recon-chess/run.py"
