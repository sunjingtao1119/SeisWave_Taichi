import taichi as ti 
import numpy as np
import matplotlib.pyplot as plt
from src.Elastic2D_CPML import ElasticWAVE
import time
ti.init(arch=ti.gpu)
nx=512
nz=512
# load model
vp =ti.field(dtype=ti.f32,shape=(nx,nz))
vp.fill(2500.) 
vs =ti.field(dtype=ti.f32,shape=(nx,nz))
vs.fill(1500.)
rho=ti.field(dtype=ti.f32,shape=(nx,nz))
rho.fill(2200)
dx=1
dz=1
accuracy=2   #  2 denotes 4-order staggered-grid
vp_max=2500  # The maximum P-wave velocity 
vs_max=1500  # The maximum S-wave velocity
# Set source and receiver points
isx=int(150) #source 
isz=int(150)
rsx=int(200) #receiver
rsz=int(200)
nt=2000
dt=0.0001
pi=np.pi
freq=60
src_scale=1
# Stability analysis
Courant_number = vp_max * dt * np.sqrt(1/dx**2 + 1/dz**2)
print(Courant_number)
if Courant_number > 1 :
    print('time step is too large, simulation will be unstable')
    exit()
# Initialize the wave field
test=ElasticWAVE(vs,vp,rho,dx,dz,dt,isx,isz,rsx,rsz,nt,accuracy,freq)  
#################### The PML boundary####################             
NPoint_Pml = 10                      # The number of grid points in PML layer
pml_x_thick=NPoint_Pml *dx;          # The thickness of PML layer in x direction
pml_z_thick=NPoint_Pml *dz;          # The thickness of PML layer in z direction
pml_parameter={}
pml_parameter["vp_max"]=vp_max                 # The maximum velocity
pml_parameter["pml_x_thick"]=pml_x_thick       # The thickness of PML layer in x direction
pml_parameter["pml_z_thick"]=pml_z_thick       # The thickness of PML layer in z direction
pml_parameter["Rcoef"]=0.000001                # The reflection coefficient         
pml_parameter["theta"]=1                        # 1 denotes implicit scheme, 0 denotes explicit scheme for the auxiliary differential equation method
pml_parameter["alpha_max_pml"]=pi*freq          # The maximum alpha value in PML layer
pml_parameter["kmax_pml"]= vs_max/(5*dx*freq)   # The maximum kappa value in PML layer            
pml_surface=[True,True,True,True]
test.SetADEPML2D(pml_surface,pml_parameter)
ts = time.time()
for i in range(nt):
    test.update_SSG_VS(i)
#ti.sync()
tend = time.time()
print(f'{tend-ts:.3} sec')

data=test.data.to_numpy()
plt.plot(test.data)
plt.show()
'''
for i in range(nt):  
    test.update_SSG_VS(i)  
    if np.mod(i,20)==0:
        plt.imshow(test.vz.to_numpy(),cmap='seismic')
        plt.clim(-1e-9,1e-9)
        plt.colorbar()
        plt.pause(0.005) 
        plt.cla()
        plt.clf()
plt.imshow(test.vz.to_numpy(),cmap='seismic')
plt.clim(-1e-9,1e-9)
plt.colorbar()
plt.show() 
'''


