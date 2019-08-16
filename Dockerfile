FROM tensorflow/tensorflow:2.0.0-beta1-gpu-py3



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
    python3-dev \
    python3 \
    python3-pip \
    libfltk1.3-dev \
    libxft-dev \
    libxinerama-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    xdg-utils 

# Python3
RUN pip3 install pip --upgrade

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


RUN apt-get install -y net-tools


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

USER ${USER_NAME}
WORKDIR ${HOME_DIR}