#FROM tensorflow/tensorflow:latest-gpu-py3

#FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

FROM tensorflow/tensorflow:latest-devel-gpu-py3


#RUN mkdir /package/
#RUN git clone https://github.com/tensorflow/tensorflow.git
 
#RUN cd "/usr/local/lib/bazel/bin" && \
#    curl -LO https://releases.bazel.build/3.0.0/release/bazel-3.0.0-linux-x86_64 && \
#    chmod +x bazel-3.0.0-linux-x86_64

#RUN cd /tensorflow_src && \
#    bazel version && \
#    git pull && \
#    git checkout v2.1.0 && \
#    ./configure && \
#    bazel build --config=cuda -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both -k //tensorflow/tools/pip_package:build_pip_package  && \
#    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /package/
# RUN pip3 install /package/*




# Python3
RUN pip3 install pip --upgrade

RUN wget https://www.dropbox.com/s/f6yp9xga5lg8323/tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl?dl=0
RUN mv tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl?dl=0 tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl



RUN apt-get update && apt-get install -y \
    build-essential \
    binutils \
    make \
    bzip2 \
    cmake \
    curl \
    git \
    g++ \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    locales \
    zlib1g-dev \
    libfltk1.3-dev \
    libxft-dev \
    libxinerama-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    xdg-utils \
    net-tools 

# Enables X11 sharing and creates user home directory
ENV USER_NAME ddangelo
ENV HOME_DIR /home/$USER_NAME
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000



RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}


ARG MPI_MAJ_VERSION="v4.0"
ARG MPI_VERSION="4.0.0"

RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/${MPI_MAJ_VERSION}/openmpi-${MPI_VERSION}.tar.gz && \
    tar zxf openmpi-${MPI_VERSION}.tar.gz && \
    cd openmpi-${MPI_VERSION} && \
    ./configure --enable-mpi-thread-multiple && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN pip3 install mpi4py
RUN pip3 install reconchess
RUN pip3 install python-chess==0.28.2
RUN pip3 install nvidia-ml-py3

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

RUN sudo apt-get install -y ssh-client
RUN sudo rm /usr/local/cuda/lib64/stubs/libcuda.so.1

#RUN export STOCKFISH_EXECUTABLE="/home/ddangelo/Dropbox/Deep Learning/recon-chess/stockfish-11-linux/Linux/stockfish_20011801_x64"
#RUN export PYTHONPATH="/home/ddangelo/Dropbox/Deep Learning/recon-chess/reconchess-strangefish"
#RUN pip3 install tqdm dataclasses


