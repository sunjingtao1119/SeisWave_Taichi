# SeisWave_Taichi
 This study presents a high-performance Python solution for 2D and 3-D parallel finite-difference seismic wave propagation that is based on the Taichi lang.
# Installation
Copy the source file directly to the computer, and then run the test2d_homegenous.py
# Requirements
Program language: Python 3.9,3.10

Hardware requirements: GEFORCE RTX 30 series GPU

Software required: 
The python library you need has Taichi (1.7.3)\
NumPy
Matplotlib
The Operating system you need Window 11 or Ubuntu.
# Usage

When using Taichi，you must first import the library，and initialize the backend
```python
 import taichi as ti
 ti.init(arch=ti.cpu)
```
Taichi can easily switch to the GPU back-end, only need to modify 1 line of code
```python
 ti.init(arch=ti.gpu)  #  ti.cuda
```
Taichi supports GPU acceleration using vulkan.
```python
 ti.init(arch=ti.vulkan)  #  ti.cuda
```
Taichi and Python share a similar syntax, but they are not identical.https://docs.taichi-lang.cn/docs/kernel_function） 
