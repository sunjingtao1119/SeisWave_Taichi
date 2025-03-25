import taichi as ti 
import numpy as np
import matplotlib.pyplot as plt
from src.Elastic2D_CPML import ElasticWAVE
from src.BaseKernel import Ricker
import time
ti.init(arch=ti.gpu)
nx=100
nz=350
# load model
vp =ti.field(dtype=ti.f32,shape=(100,350))
vp.fill(1500.)
vs =ti.field(dtype=ti.f32,shape=(100,350))
vs.fill(1000.)
rho=ti.field(dtype=ti.f32,shape=(100,350))
rho.fill(2200.)
dx=1.
dz=1.
dt=1e-4
accuracy=3  #  3 denotes 6th-order staggered-grid
isx=int(3)    #source 
isz=int(175)
rsx=int(nx/2-35) #receiver
rsz=int(nz/2-35)
nt=2000
dt=1e-4
pi=np.pi
freq=25 
src_scale=1000
pi=np.pi
test=ElasticWAVE(vs,vp,rho,dx,dz,dt,isx,isz,rsx,rsz,nt,accuracy,freq)
#################### The PML boundary#################### 
vp_max=2000
vs_max=1000                  
NPoint_Pml = 10                      # The number of grid points in PML layer
pml_x_thick=NPoint_Pml *dx;          # The thickness of PML layer in x direction
pml_z_thick=NPoint_Pml *dz;          # The thickness of PML layer in z direction
pml_parameter={}
pml_parameter["vp_max"]=vp_max                # The maximum velocity
pml_parameter["pml_x_thick"]=pml_x_thick     # The thickness of PML layer in x direction
pml_parameter["pml_z_thick"]=pml_z_thick     # The thickness of PML layer in z direction
pml_parameter["Rcoef"]=0.000001              # The reflection coefficient         
pml_parameter["theta"]=1                     # 1 denotes implicit scheme, 0 denotes explicit scheme for the auxiliary differential equation method
pml_parameter["alpha_max_pml"]=pi*100        # The maximum alpha value in PML layer
pml_parameter["kmax_pml"]= vs_max/(5*dx*100)   # The maximum kappa value in PML layer            
pml_surface=[False,True,True,True]
test.SetADEPML2D(pml_surface,pml_parameter)
'''
ts = time.time()
for i in range(nt):
    test.update(i)
tend = time.time()
print(f'{tend-ts:.3} sec')
'''
test.AEA_init()            # freedom boudary condition
for i in range(nt):
    src=Ricker(i,dt,freq,src_scale) 
    test.AEA()              # freedom boudary condition
    test.update_SSG_VS(i,src)   
    if np.mod(i,20)==0:
        plt.imshow(test.vx.to_numpy(),cmap='seismic')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.colorbar()
        plt.clim(-5e-3,5e-3)
        plt.pause(0.05)
        plt.cla()
        plt.clf()
plt.imshow(test.vx.to_numpy(),cmap='seismic')
plt.colorbar()
plt.clim(-5e-3,5e-3)
plt.show() 