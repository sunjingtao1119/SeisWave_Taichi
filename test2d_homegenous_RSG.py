import taichi as ti 
import numpy as np
import matplotlib.pyplot as plt
from src.Elastic2D_RSG_CPML import ElasticWAVE
import time
from scipy.io import savemat
ti.init(arch=ti.gpu)
nx=241
nz=241
# load model
vp =ti.field(dtype=ti.f32,shape=(nx,nz))
vp.fill(2000.) 
vs =ti.field(dtype=ti.f32,shape=(nx,nz))
vs.fill(1150.)
rho=ti.field(dtype=ti.f32,shape=(nx,nz))
rho.fill(2000)
dx=2
dz=2
accuracy=2   #  4 denotes 8th-order staggered-grid
vp_max=2000 
vs_max=1150  
# Set source and receiver points
isx=int(nx/2)    #source 
isz=int(nz/2)
rsx=int(nx/2+40) #receiver
rsz=int(nz/2+40)
nt=801
dt=3e-4
pi=np.pi
freq=40
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
            
NPoint_Pml = 15                      # The number of grid points in PML layer
pml_x_thick=(NPoint_Pml-1) *dx;          # The thickness of PML layer in x direction
pml_z_thick=(NPoint_Pml-1) *dz;          # The thickness of PML layer in z direction
pml_parameter={}
pml_parameter["vp_max"]=vp_max                 # The maximum velocity
pml_parameter["pml_x_thick"]=pml_x_thick     # The thickness of PML layer in x direction
pml_parameter["pml_z_thick"]=pml_z_thick     # The thickness of PML layer in z direction
pml_parameter["Rcoef"]=0.000001              # The reflection coefficient         
pml_parameter["theta"]=1                     # 1 denotes implicit scheme, 0 denotes explicit scheme for the auxiliary differential equation method
pml_parameter["alpha_max_pml"]=pi*freq        # The maximum alpha value in PML layer
pml_parameter["kmax_pml"]= vs_max/(5*dx*freq)   # The maximum kappa value in PML layer            
pml_surface=[True,True,True,True]
test.SetADEPML2D(pml_surface,pml_parameter)
################### Initial Source ##################### 
src_type=2      # The type of source
src_scale=1.
t0=1/test.f0
test.init_src(src_type,src_scale)
########################################################

ts = time.time()
for i in range(nt):
    test.update_RSG(i)
tend = time.time()
print(f'{tend-ts:.3} sec')
data=test.data.to_numpy()
plt.plot(test.data)
plt.show()
file_name = 'data2d_rsg4.mat'
savemat(file_name, {'data2d_rsg4': data})   
 
'''
for i in range(nt):  
    test.update_RSG(i) 
    if np.mod(i,20)==0:
        plt.imshow(test.vz.to_numpy(),cmap='seismic')
        plt.clim(-1e-8,1e-8)
        plt.colorbar()
        plt.pause(0.005) 
        plt.cla()
        plt.clf()
plt.imshow(test.vz.to_numpy(),cmap='seismic')
plt.clim(-1e-8,1e-8)
plt.colorbar()
plt.show() 
'''



