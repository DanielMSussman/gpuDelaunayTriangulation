# INSTALLATION {#install}

A general CMake scheme is included with the repository. 

## Basic compilation

* cd into build/ directory. 
* $ cmake ..
* $ make

# Requirements

The current iteration of the code was written using some features of C++11, and was compiled using CUDA-11.01, and gcc/g++-6 but does not use advanced CUDA features. It was tested on a compute_35 GPU.

Default compilation is via CMake, so you need that, too; comparisons between existing triangulation schemes currently use CGAL, using the header-only version 5.0.2

# Operating systems tested

This code has been compiled and run on Ubuntu 18.04, as well as Windows Subsystems for Linux 2

# Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads
CMAKE: https://cmake.org
