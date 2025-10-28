import taichi as ti 
import numpy as np
import matplotlib.pyplot as plt
from src.Elastic3D_SSG_Tensor import ElasticWAVE
from  src.BaseKernel import Ricker, Ricker2
from scipy.io import savemat

import time
ti.init(arch=ti.vulkan)
nx=121
ny=121
nz=121
# load model
vp =ti.field(dtype=ti.f32,shape=(nx,ny,nz))
vp.fill(3000.)
vs =ti.field(dtype=ti.f32,shape=(nx,ny,nz))
temp=3000./1.67
vs.fill(temp)
rho=ti.field(dtype=ti.f32,shape=(nx,ny,nz))
rho.fill(2500)
dx=2.5
dy=2.5
dz=2.5
nt=801
accuracy=4        #  3 denotes 6th-order staggered-grid
vp_max=3000
vs_max=temp 
# 接收点位置
rsx_np=np.ones((1,81),dtype=int)
rsx_np=21*rsx_np
rsy_np=np.ones((1,81),dtype=int)
rsy_np=21*rsy_np
rsz_np=np.linspace(21,101,81,dtype=int)
rsz_np=rsz_np.reshape(1,81)
rsx=ti.field(dtype=int,shape=(1,81))
rsx.from_numpy(rsx_np)
rsy=ti.field(dtype=int,shape=(1,81))
rsy.from_numpy(rsy_np)
rsz=ti.field(dtype=int,shape=(1,81))    
rsz.from_numpy(rsz_np)
# 震源设置
isx=int(61)  # 震源位置
isy=int(61)
isz=int(101)
dt=0.0003
pi=np.pi
freq=60
src_scale=1
src=ti.field(dtype=ti.f32,shape=(nt))   # intialize the source wavelet
# Generate Ricker wavelet
Ricker2(src,nt,dt,freq,src_scale)
# Moment tensor source implementation
MT=ti.field(dtype=ti.f32,shape=((3,3)))
MT[0,0]=0
MT[0,1]=ti.sqrt(1/2)
MT[0,2]=0
MT[1,0]=ti.sqrt(1/2)
MT[1,1]=0
MT[1,2]=0
MT[2,0]=0
MT[2,1]=0
MT[2,2]=0 
# Stability analysis
Courant_number = vp_max * dt * np.sqrt(1/dx**2 + 1/dy**2+1/dz**2)
print(Courant_number)
if Courant_number > 1 :
    print('time step is too large, simulation will be unstable')
    exit()
# Initialize the wave field
test=ElasticWAVE(vs,vp,rho,dx,dy,dz,dt,isx,isy,isz,rsx,rsy,rsz,MT,src_scale,nt,accuracy,freq)
#################### The PML boundary####################                  
NPoint_Pml = 15                      # The number of grid points in PML layer
pml_x_thick=NPoint_Pml *dx;          # The thickness of PML layer in x direction
pml_y_thick=NPoint_Pml *dy;  
pml_z_thick=NPoint_Pml *dz;          # The thickness of PML layer in z direction
pml_parameter={}
pml_parameter["vp_max"]=vp_max                # The maximum velocity
pml_parameter["pml_x_thick"]=pml_x_thick     # The thickness of PML layer in x direction
pml_parameter["pml_y_thick"]=pml_y_thick     # The thickness of PML layer in y direction
pml_parameter["pml_z_thick"]=pml_z_thick     # The thickness of PML layer in z direction
pml_parameter["Rcoef"]=0.000001              # The reflection coefficient         
pml_parameter["theta"]=1                     # 1 denotes implicit scheme, 0 denotes explicit scheme for the auxiliary differential equation method
pml_parameter["alpha_max_pml"]=pi*freq        # The maximum alpha value in PML layer
pml_parameter["kmax_pml"]= vs_max/(5*dx*freq)   # The maximum kappa value in PML layer            
pml_surface=[True,True,True,True,True,True]  # The PML boundary condition in x,y,z direction
test.SetADEPML3D(pml_surface,pml_parameter)

ts = time.time()
for i in range(nt):    
    test.update_SSG(i,src[i])
ti.sync()
tend = time.time()    
print(f'{tend-ts:.3} sec')
data=test.data.to_numpy()
data=data[:,0]
file_name = 'data.mat'
savemat(file_name,{'data': data})
#wigb(data)
plt.plot(data)
plt.show()
'''
ts = time.time()

for i in range(nt):
    test.update_SSG(i)
    if np.mod(i,20)==0:
        im=test.vx.to_numpy()
        plt.imshow(im[:,isy,:] ,cmap='seismic')  #[isx,:,:]  [:,isx,:] [:,:,isx]
        plt.colorbar()
        plt.clim(-1e-10,1e-10)
        plt.pause(0.01)
        plt.cla()
        plt.clf()
#print(test.k_z)
plt.imshow(im[:,isy,:],cmap='seismic')
plt.colorbar()
plt.clim(-1e-10,1e-10)
plt.show()
'''

