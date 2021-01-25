## 1 - Install Singularity for running the container

The following instructions were tested on Ubuntu 16.04 for Singularity version 2.6.1. More detailed installation steps are given in [https://sylabs.io/guides/2.6/user-guide/installation.html](https://sylabs.io/guides/2.6/user-guide/installation.html)

```shell
sudo apt-get update && sudo apt-get install python dh-autoreconf build-essential libarchive-dev
VER=2.6.1
wget https://github.com/sylabs/singularity/releases/download/$VER/singularity-$VER.tar.gz
tar xvf singularity-$VER.tar.gz
cd singularity-$VER
./configure --prefix=/usr/local --sysconfdir=/etc
make
sudo make install
```
