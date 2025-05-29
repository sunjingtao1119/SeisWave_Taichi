import taichi as ti 
import numpy as np
import matplotlib.pyplot as plt
from src.Elastic3D_RSG_EAL import ElasticWAVE
from src.BaseKernel import Ricker
import time
ti.init(arch=ti.gpu)
# load model
nx=170
ny=170
nz=170
# load model
vp =ti.field(dtype=ti.f32,shape=(nx,ny,nz))
vp.fill(2000)
vs =ti.field(dtype=ti.f32,shape=(nx,ny,nz))
vs.fill(1150)
rho=ti.field(dtype=ti.f32,shape=(nx,ny,nz))
rho.fill(2000)
vp_max=2000
vs_max=1150  
dx=1
dy=1
dz=1
nt=1500
isx=int(nx/2)
isy=int(ny/2)
isz=int(nz/2)
rsx=int(nx/2+20)
rsy=int(ny/2+20)
rsz=int(nz/2+20)
dt=1e-4
accuracy=5      #  3 denotes 6th-order staggered-grid
pi=np.pi
freq=30
src_scale=1000
# Stability analysis

# Initialize the wave   
test=ElasticWAVE(vs,vp,rho,dx,dy,dz,dt,isx,isy,isz,rsx,rsy,rsz,nt,accuracy,freq)
#################### The PML boundary####################              
NPoint_Pml=25                    # The number of grid points in PML layer
pml_x_thick=NPoint_Pml *dx;          # The thickness of PML layer in x direction
pml_y_thick=NPoint_Pml *dy;  
pml_z_thick=NPoint_Pml *dz;          # The thickness of PML layer in z direction
pml_parameter={}  
pml_parameter["vp_max"]=vp_max                 # The maximum velocity
pml_parameter["pml_x_thick"]=pml_x_thick       # The thickness of PML layer in x direction
pml_parameter["pml_y_thick"]=pml_y_thick       #   thickness of PML layer in y direction
pml_parameter["pml_z_thick"]=pml_z_thick       # The thickness of PML layer in z direction
pml_parameter["Rcoef"]=0.000001                # The reflection coefficient         
pml_parameter["theta"]=1                       # 1 denotes implicit scheme, 0 denotes explicit scheme for the auxiliary differential equation method
pml_parameter["alpha_max_pml"]=pi*freq         # The maximum alpha value in PML layer
pml_parameter["kmax_pml"]= vs_max/(5*dx*freq)  # The maximum kappa value in PML layer            
pml_surface=[True,True,True,True,True,True]    # The PML boundary condition in x,y,z direction
test.SetADEPML3D(pml_surface,pml_parameter)
#test.SetEAL3D(pml_surface,pml_parameter)

'''
for i in range(nt):
    src=Ricker(i,dt,freq,src_scale)
    test.update_RSG(i,src)

im=test.vz.to_numpy()
data=test.data
#plt.imshow(data ,cmap='seismic')
plt.plot(data)
plt.show()

''' 

for i in range(nt):
    src=Ricker(i,dt,freq,src_scale)
    test.update_RSG(i,src)
    if np.mod(i,20)==0:
        im=test.vz.to_numpy()
        plt.imshow(im[:,isx,:] ,cmap='seismic')  #[isx,:,:]  [:,isx,:] [:,:,isx]
        plt.colorbar()
        plt.clim(-0.5e-5,0.5e-5)
        plt.pause(0.01)
        plt.cla()
        plt.clf()
#print(test.k_z)
plt.imshow(im[:,isx,:],cmap='seismic')
data=im[:,isx,:]
plt.colorbar()
plt.clim(-0.5e-5,0.5e-5)
plt.show()

 