[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

## <b>GEOBIT: Geodesic Binary Descriptor for Nonrigid RGB-D Images</b> <br>[[Project Page]](https://www.verlab.dcc.ufmg.br/descriptors/iccv2019/) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Nascimento_GEOBIT_A_Geodesic-Based_Binary_Descriptor_Invariant_to_Non-Rigid_Deformations_for_ICCV_2019_paper.html)

<img src='images/geobit.png' align="center" width=900 />

Code repository for the paper  "GEOBIT: A Geodesic-Based Binary Descriptor Invariant to Non-Rigid Deformations for RGB-D Images", presented in ICCV 2019. GeoBit is a handcrafted binary descriptor that combines appearance and geometric information from RGB-D images to handle isometric non-rigid deformations. It leverages geodesic isocurve information, from heat flow in the surface manifold, to select the feature binary tests.

If you find this code useful for your research, please cite the paper:

```
@inproceedings{erickson2019iccv,
author = {Erickson R. Nascimento and Guilherme Potje and Renato Martins and Felipe Chamone and Mario F. M. Campos and Ruzena Bajcsy},
title = {{GEOBIT}: A Geodesic-Based Binary Descriptor Invariant to Non-Rigid Deformations for {RGB-D} Images},
booktitle={IEEE International Conference on Computer Vision (ICCV)},
year = {2019}
}
```

## Installation

- Install Dependencies
  
    **OpenCV**
  
    ```bash
    # Install minimal prerequisites (Ubuntu 18.04 as reference)
    sudo apt update && sudo apt install -y cmake g++ wget unzip
    # Download and unpack sources
    wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
    unzip opencv.zip
    unzip opencv_contrib.zip
    # Create build directory and switch into it
    mkdir -p build && cd build
    # Configure
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master
    # Build
    cmake --build .
    ```
    **PCL**
    
    ```bash
    sudo apt install libpcl-dev
    ```
    **Suitsparce**

    ```bash
    sudo apt-get install python-software-properties
    sudo add-apt-repository ppa:jmaye/ethz
    sudo apt-get update
    sudo apt-get install libsuitesparse-dev
    ```
- Compiling

```bash
mkdir build
cd build
cmake ..
```
- Testing

```bash
./nonrigid_descriptor -inputdir ../example -refcloud cloud_1 -clouds cloud_1 -datasettype real
```

## RGB-D Dataset

<table>
<tr>
<td align="center"><img src="images/geobit_gif1.gif" ></td>
<td align="center"><img src="images/geobit_gif2.gif" ></td>
<td align="center"><img src="images/geobit_gif3.gif" ></td>
</tr>
<tr>
<td align="center"><img src="images/geobit_gif4.gif" ></td>
<td align="center"><img src="images/geobit_gif5.gif" ></td>
<td align="center"><img src="images/geobit_gif6.gif" ></td>
</tr>
<tr>
<td align="center"><img src="images/geobit_gif7.gif" ></td>
<td align="center"><img src="images/geobit_gif8.gif" ></td>
<td align="center"><img src="images/geobit_gif9.gif" ></td>
</tr>
</table>

## Getting Started

### Institution ###

Universidade Federal de Minas Gerais (UFMG)\
Department of Computer Science\
Belo Horizonte - Minas Gerais - Brazil 


### Laboratory ###

![VeRLab](https://www.dcc.ufmg.br/dcc/sites/default/files/public/verlab-logo.png)

**VeRLab:** Laboratory of Computer Vison and Robotics
https://www.verlab.dcc.ufmg.br
