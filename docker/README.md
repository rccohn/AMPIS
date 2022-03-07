container: rccohn/ampis

Image for a consistent environment to run AMPIS in.
Note: for working with GPU's, a compatible nvidia gpu driver and the [nvidia container runtime](https://github.com/NVIDIA/nvidia-container-runtime) must be installed.

./container provides convenient commands for building, running, and removing the image.
./container -b to build, ./container -R to run with cpu, ./container -r to run with GPU.

To run a jupyter notebook in docker:
$ docker run --init --rm -itd -v $(pwd):/path/in/container \
             -p 8888:8888
             --gpus all `#optional, only if you have gpu runtime`\
             rccohn/ampis 
             
Inside the container: run
$ python -m jupyterlab `# start jupyter` \
	--no-browser `# browser cannot run inside container, access from host`\
        --ip localhost `#host ip` \
        --port 8888 \
	--NotebookApp.token='' `# disable token (local container -> no security threat)`\
	--NotebookApp.password='' `# disable password (local container -> no security threat)` \
	--allow-root `# allow jupyter to run with root account (of container)` \
	--notebook-dir /path/to/start/jupyter/in

