import taichi as ti 
import numpy as np
import matplotlib.pyplot as plt
from src.Elastic3D_SSG_CPML import ElasticWAVE
from  src.BaseKernel import Ricker, Ricker2
import time
from scipy.io import savemat
#ti.init(arch=ti.cpu,cpu_max_num_threads=20)  # 215-232s   171# 106s
ti.init(arch=ti.cpu)
nx=171
ny=171
nz=171
# load model
vp =ti.field(dtype=ti.f32,shape=(nx,ny,nz))
vp.fill(4500)
vs =ti.field(dtype=ti.f32,shape=(nx,ny,nz))
temp=2600
vs.fill(temp)
rho=ti.field(dtype=ti.f32,shape=(nx,ny,nz))
rho.fill(2730)
dx=1.5
dy=1.5
dz=1.5
nt=1000
vp_max=4500
vs_max=temp 
isx=int(nx/2)
isy=int(ny/2)
isz=int(nz/2)
rsx=int(nx/2+40)
rsy=int(ny/2+40)
rsz=int(nz/2+40)
dt=0.0001
accuracy=5       #  The accuracy order of the spatial derivative
pi=np.pi
freq=30         # dominant frequency of the source wavelet
src_scale=1      # source scale
src=ti.field(dtype=ti.f32,shape=(nt))   # intialize the source wavelet
# Generate Ricker wavelet
Ricker(src,nt,dt,freq,src_scale)
# Stability analysis
Courant_number = vp_max * dt * np.sqrt(1/dx**2 + 1/dy**2+1/dz**2)
print(Courant_number)
if Courant_number > 1 :
    print('time step is too large, simulation will be unstable')
    exit()
# Initialize the wave field
test=ElasticWAVE(vs,vp,rho,dx,dy,dz,dt,isx,isy,isz,rsx,rsy,rsz,nt,accuracy,freq) 
#################### The PML boundary####################                
NPoint_Pml = 15                                 # The number of grid points in PML layer
pml_x_thick=NPoint_Pml *dx;                     # The thickness of PML layer in x direction
pml_y_thick=NPoint_Pml *dy;  
pml_z_thick=NPoint_Pml *dz;                     # The thickness of PML layer in z direction
pml_parameter={}
pml_parameter["vp_max"]=vp_max                  # The maximum velocity
pml_parameter["pml_x_thick"]=pml_x_thick        # The thickness of PML layer in x direction
pml_parameter["pml_y_thick"]=pml_y_thick        # The thickness of PML layer in y direction
pml_parameter["pml_z_thick"]=pml_z_thick        # The thickness of PML layer in z direction
pml_parameter["Rcoef"]=0.000001                 # The reflection coefficient         
pml_parameter["theta"]=1                        # 1 denotes implicit scheme, 0 denotes explicit scheme for the auxiliary differential equation method
pml_parameter["alpha_max_pml"]=pi*freq          # The maximum alpha value in PML layer
pml_parameter["kmax_pml"]= vs_max/(5*dx*freq)   # The maximum kappa value in PML layer            
pml_surface=[True,True,True,True,True,True]     # The PML boundary condition in x,y,z direction
test.SetADEPML3D(pml_surface,pml_parameter)     # Iinital the PML parameters
#################### The PML boundary#################### 
ts = time.time()
for i in range(nt):
    test.update_SSG(i,src[i])
ti.sync()
tend = time.time()
print(f'{tend-ts:.3} sec')
data=test.data.to_numpy()
file_name = 'data.mat'
#savemat(file_name, {'data': data})
plt.plot(test.data)
plt.show()
'''
ts = time.time()
for i in range(nt):
    test.update_SSG(i,src[i])
    if np.mod(i,10)==0:
        im=test.vz.to_numpy()
        plt.imshow(im[:,isx,:] ,cmap='seismic',interpolation='bilinear')  #[isx,:,:]  [:,isx,:] [:,:,isx]
        plt.colorbar()
        plt.clim(-5e-10,5e-10)
        plt.cla()
        plt.clf()

im=test.vz.to_numpy()
#print(test.k_z)
plt.imshow(im[:,isx,:],cmap='seismic',interpolation='bilinear')
plt.colorbar()
plt.clim(-5e-10,5e-10)
plt.show()
'''

