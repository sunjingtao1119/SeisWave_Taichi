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
vp.fill(2000.) 
vs =ti.field(dtype=ti.f32,shape=(nx,nz))
vs.fill(1150.)
rho=ti.field(dtype=ti.f32,shape=(nx,nz))
rho.fill(2000)
dx=1
dz=1
accuracy=2   #  3 denotes 6th-order staggered-grid
vp_max=2000 
vs_max=1150   
# Set source and receiver points
isx=int(240/2)    #source 
isz=int(240/2)
rsx=int(240/2+45) #receiver
rsz=int(240/2+45)
nt=2000
dt=1e-3
pi=np.pi
freq=100 
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
pml_parameter["pml_x_thick"]=pml_x_thick     # The thickness of PML layer in x direction
pml_parameter["pml_z_thick"]=pml_z_thick     # The thickness of PML layer in z direction
pml_parameter["Rcoef"]=0.000001              # The reflection coefficient         
pml_parameter["theta"]=1                     # 1 denotes implicit scheme, 0 denotes explicit scheme for the auxiliary differential equation method
pml_parameter["alpha_max_pml"]=pi*freq        # The maximum alpha value in PML layer
pml_parameter["kmax_pml"]= vs_max/(5*dx*freq)   # The maximum kappa value in PML layer            
pml_surface=[True,True,True,True]
test.SetADEPML2D(pml_surface,pml_parameter)

ts = time.time()
for i in range(nt):
    test.update_SSG_VS(i)
tend = time.time()
print(f'{tend-ts:.3} sec')

plt.plot(test.data)
plt.show()
'''
data=test.data.to_numpy()

for i in range(nt):  
    src=Ricker(i,dt,freq,src_scale) 
    test.update_SSG_SV(nt,src)   
    if np.mod(i,20)==0:
        plt.imshow(test.vx.to_numpy(),cmap='seismic')
        plt.clim(-1e-3,1e-3)
        plt.colorbar()
        plt.pause(0.005) 
        plt.cla()
        plt.clf()
plt.imshow(test.vz.to_numpy(),cmap='seismic')
plt.clim(-1e-5,1e-5)
plt.colorbar()
plt.show() 
'''
#data=test.vx.to_numpy()

# 数值实验表明：分裂波场的有限差分的时域稳定性大于非分离波场


