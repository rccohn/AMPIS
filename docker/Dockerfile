# OS and architecture, needed for base images and nvidia key rotation
# operating system
ARG dist_os=ubuntu
# version major
ARG version_major=18
# version minor
ARG version_minor=04
# architecture
ARG arch=x86_64
ARG distro=${dist_os}${version_major}${version_minor}

# docker CUDA version differs from cuda used in torch --> this is ok as cudatoolkit is 
# a dependency of pytorch. Using latest docker container gives better security.
FROM nvidia/cuda:11.6.0-base-${dist_os}${version_major}.${version_minor} AS base
# build arguments: non-root user
ARG USER=ampis

# need to re-declare these args or else they will disappear (new build stage)
ARG distro
ARG arch

# Set the default shell for RUN commands
SHELL ["/bin/bash", "-c"]

# Set character encoding environment variables
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Allow apt-get install without interaction from console
ENV DEBIAN_FRONTEND=noninteractive

# nvidia key rotation (security update, needs to be applied or else 
# apt-get update will not work) 
# for more information see:
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# Note that their 'recommended' approach does not seem to work in Docker (runs into dpkg issues)
# but their "not recommended" approach for manually installing the key works...

RUN apt-key del "7fa2af80" \
    && apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub"


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

# add non root user
RUN groupadd --gid=1000 ${USER} \
    && useradd --uid 1000 --gid 1000 -m ${USER}

USER ${USER}

ENV HOME=/home/ampis

WORKDIR ${HOME}

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
	rm Miniconda3-latest-Linux-x86_64.sh

# Set the path env to inclcude miniconda
ENV PATH ${HOME}/miniconda/bin:$PATH

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
ENV PATH="${HOME}/miniconda/envs/env/bin:${PATH}"

# installing pip dependencies with options doesn't seem to work in conda yaml
# so we have an additional requirements file for detectron2 and AMPIS
COPY requirements.txt requirements.txt
# requirements file does not like -f, so we run install command directly
RUN pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
  
RUN pip install -r requirements.txt
