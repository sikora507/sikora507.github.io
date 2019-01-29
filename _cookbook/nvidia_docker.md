---
layout: post
title:  "Install nvidia-docker"
date:   2019-01-27 11:53:35 +0100
categories: nvidia docker linux
---
One very convenient way to do experiments with Tensorflow or Keras on linux is to use docker with GPU support.

Only prerequisite is to have
[nvidia graphics driver installed](/cookbook/linux_nvidia_driver).

For more details, you can check this website <https://www.tensorflow.org/install/docker> and follow other links up there, but here is the TLDR; version.

## Preparation
1. Remove old docker versions in case you have them:
```
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```

Now here's the short version of what is on <https://docs.docker.com/install/linux/docker-ce/ubuntu/>:

## Adding repository
1. Update the `apt` package index:
```
$ sudo apt-get update
```
2. Install packages to allow apt to use a repository over HTTPS:
```
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```
3. Add Docker’s official GPG key:
```
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
4. Set up **stable** repository:
```
$ sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
```

## Install Docker CE
Docker CE is the regular *normal* docker, now it's called Docker CE. It's required to install nvidia-docker environment afterwards.

1. Update the `apt` package index:
```
$ sudo apt-get update
```
2. Install the latest version of Docker CE:
```
$ sudo apt-get install docker-ce
```
3. Verify that Docker CE is installed correctly by running the hello-world image:
```
$ sudo docker run hello-world
```
You should see output similar to:
![Docker Hello World!](/assets/img/docker_hello_world.png)

## Install nvidia-docker
Here's in short what is on original website <https://github.com/NVIDIA/nvidia-docker>:
1. If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers:
```
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
```
2. Add the package repositories:
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```
3. Install nvidia-docker2 and reload the Docker daemon configuration:
```
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```
4. Test nvidia-smi with the latest official CUDA image:
```
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```
You should see output similar to:
![Docker Hello World!](/assets/img/docker_nvidia_test.png)