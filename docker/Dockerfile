# NOTE due to nvidia key update, apt-get update command does not work
# key needs to be rotated before the container can be built.
# This requires multiple changes in the Dockerfile, so I will have to
# get to this when I have time.
# more info: 
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
FROM nvidia/cuda:11.1.1-base-ubuntu18.04

# Set the default shell
SHELL ["/bin/bash", "-c"]

# Set character encoding environment variables
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Allow apt-get install without interaction from console
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /root
WORKDIR /root/

# System updates and configurations
RUN apt-get update && apt-get -y --no-install-recommends install \
		ca-certificates \
		git \
		ssh \
		wget \ 
		build-essential \
		libglib2.0-0 `#cv2 dependencies`\
		libsm6 \
		libxrender1 \
		libxext6 && \
		apt-get clean  && \
		apt-get autoremove && \
		rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
	rm Miniconda3-latest-Linux-x86_64.sh

# Set the path env to inclcude miniconda
ENV PATH /root/miniconda/bin:$PATH

# copy env files to root
COPY env.yml env.yml

# Install venv
# env.yml specifies environment name as "env"
# note: the default env.yml includes several large libraries 
#       cuda, pytorch, etc
#      For adding smaller packages, adding another 
#      RUN pip install command will be MUCH faster
#      than altering env.yml as Docker can use the 
#      existing layers for these larger libraries.
# for now, the environment install is split among this step
# and the next RUN pip install command for this reason
RUN conda env create --file env.yml
# Set conda environment as default pyhton environment (ie no need to source activate)
ENV PATH="/root/miniconda/envs/env/bin:${PATH}"

# installing pip dependencies with options doesn't seem to work in conda yaml
# so we have an additional requirements file for detectron2 and AMPIS
COPY requirements.txt requirements.txt
# requirements file does not like -f, so we run install command directly
RUN pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
  
RUN pip install -r requirements.txt

WORKDIR /home
