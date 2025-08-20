import taichi as ti
import numpy as np
from src.Differential3D_RSG import Drd1bm,Drd2bm,Drd3bm,Drd4bm,Drd1fm,Drd2fm,Drd3fm,Drd4fm
from src.BaseFun import Ricker2
pi=np.pi

@ti.data_oriented
class ElasticWAVE:
    def __init__(self,
                 vs:ti.field, 
                 vp:ti.field,
                 rho:ti.field,
                 dx:float,
                 dy:float,
                 dz:float,
                 dt:float,
                 isx:int,
                 isy:int,
                 isz:int,
                 rsx:int,
                 rsy:int,
                 rsz:int,
            MT:ti.field, 
                 src_scale:float, 
                 nt:int,               
                 accuracy:int,
                 freq=100,
                 fieldtype=ti.f32):
        # Initialize model parameters
        self.vs=vs
        self.vp=vp
        self.rho=rho
        self.gridsize=vs.shape
        self.dx=dx
        self.dy=dy
        self.dz=dz
        self.xmin=0*dx
        self.ymin=0*dy
        self.zmin=0*dz
        self.star=accuracy
        self.c=self.diff_coff(accuracy)
        self.xmax=dx*(self.gridsize[1]-1-2*accuracy)-self.xmin    # Calculate the maximum x-coordinate based on grid size and spacing
        self.ymax=dy*(self.gridsize[2]-1-2*accuracy)-self.ymin    # Calculate the maximum z-coordinate based on grid size and spacing
        self.zmax=dz*(self.gridsize[0]-1-2*accuracy)-self.zmin    # Calculate the maximum z-coordinate based on grid size and spacing
        self.xmin_EAL=5*dx
        self.ymin_EAL=5*dy
        self.zmin_EAL=5*dz
        self.star=accuracy
        self.c=self.diff_coff(accuracy)
        self.xmax_EAL=dx*(self.gridsize[1]-1-2*accuracy)-self.xmin_EAL    # Calculate the maximum x-coordinate based on grid size and spacing
        self.ymax_EAL=dy*(self.gridsize[2]-1-2*accuracy)-self.ymin_EAL    # Calculate the maximum z-coordinate based on grid size and spacing
        self.zmax_EAL=dz*(self.gridsize[0]-1-2*accuracy)-self.zmin_EAL    # Calculate the maximum z-coordinate based on grid size and spacing


        self.mu=self.Compute_mu(fieldtype)
        self.lam=self.Compute_lam(fieldtype)
        # source term
        self.f0=freq
        self.dt=dt
        self.isx=isx
        self.isy=isy
        self.isz=isz
        self.rsx=rsx
        self.rsy=rsy
        self.rsz=rsz
        self.MT=MT
        self.src_scale=src_scale
        datasize=rsx.shape[1]
        self.datasize=datasize
        self.data=ti.field(dtype=ti.f32,shape=(nt,datasize))
        # velocity field and stress field initial 
        self.vx =ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vy =ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vz =ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxy=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.syy=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.syz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.szz=ti.field(dtype=fieldtype,shape=self.gridsize)
        # Initialize PML (Perfectly Matched Layer) parameters
        self.pml_x       =ti.field(fieldtype,shape=self.gridsize[1])
        self.pml_x_half  =ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.alpha_x     =ti.field(fieldtype,shape=self.gridsize[1])
        self.alpha_x_half=ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.k_x         =ti.field(fieldtype,shape=self.gridsize[1])
        self.k_x_half    =ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.b_x         =ti.field(fieldtype,shape=self.gridsize[1])
        self.b_x_half    =ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.a_x         =ti.field(fieldtype,shape=self.gridsize[1])
        self.a_x_half    =ti.field(fieldtype,shape=self.gridsize[1])   # half grid

        self.pml_y       =ti.field(fieldtype,shape=self.gridsize[2])
        self.pml_y_half  =ti.field(fieldtype,shape=self.gridsize[2])   # half grid
        self.alpha_y     =ti.field(fieldtype,shape=self.gridsize[2])
        self.alpha_y_half=ti.field(fieldtype,shape=self.gridsize[2])   # half grid
        self.k_y        =ti.field(fieldtype,shape=self.gridsize[2])
        self.k_y_half    =ti.field(fieldtype,shape=self.gridsize[2])   # half grid
        self.b_y         =ti.field(fieldtype,shape=self.gridsize[2])
        self.b_y_half    =ti.field(fieldtype,shape=self.gridsize[2])   # half grid
        self.a_y         =ti.field(fieldtype,shape=self.gridsize[2])
        self.a_y_half    =ti.field(fieldtype,shape=self.gridsize[2])   # half grid


        self.pml_z       =ti.field(fieldtype,shape=self.gridsize[0])
        self.pml_z_half  =ti.field(fieldtype,shape=self.gridsize[0])
        self.alpha_z     =ti.field(fieldtype,shape=self.gridsize[0])
        self.alpha_z_half=ti.field(fieldtype,shape=self.gridsize[0])
        self.k_z         =ti.field(fieldtype,shape=self.gridsize[0])
        self.k_z_half    =ti.field(fieldtype,shape=self.gridsize[0])
        self.b_z         =ti.field(fieldtype,shape=self.gridsize[0])
        self.b_z_half    =ti.field(fieldtype,shape=self.gridsize[0])
        self.a_z         =ti.field(fieldtype,shape=self.gridsize[0])
        self.a_z_half    =ti.field(fieldtype,shape=self.gridsize[0])
        self.xmin_pml=self.xmin
        self.xmax_pml=self.xmax
        self.zmin_pml=self.zmin
        self.zmax_pml=self.zmax
        # Initialize memory variables for PML (Perfectly Matched Layer)
        self.memory_sxx_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_sxz_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_sxy_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_syy_dy=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_sxy_dy=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_syz_dy=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_szz_dz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_sxz_dz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_syz_dz=ti.field(dtype=fieldtype,shape=self.gridsize)
            
        self.memory_dvx_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvy_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvz_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvx_dy=ti.field(dtype=fieldtype,shape=self.gridsize) 
        self.memory_dvy_dy=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvz_dy=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvx_dz=ti.field(dtype=fieldtype,shape=self.gridsize) 
        self.memory_dvy_dz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvz_dz=ti.field(dtype=fieldtype,shape=self.gridsize)

    def Compute_mu(self,fieldtype):
        size=self.vs.shape
        vs_np=self.vs.to_numpy()
        rho_np=self.rho.to_numpy()
        mu_np=rho_np*vs_np**2
        mu=ti.field(dtype=fieldtype,shape=size)
        mu.from_numpy(mu_np)
        return mu

    def Compute_lam(self,fieldtype):
        size=self.vs.shape
        vs_np=self.vs.to_numpy()
        vp_np=self.vp.to_numpy()
        rho_np=self.rho.to_numpy()
        lam_np=rho_np*(vp_np**2-2*vs_np**2)
        lam=ti.field(dtype=fieldtype,shape=size)
        lam.from_numpy(lam_np)
        return lam

    @ti.kernel
    def update_RSG(self,nt:int):
        star=self.star
        dx =self.dx
        dy =self.dy
        dz =self.dz
        dt =self.dt
        isz=self.isz
        isy=self.isy
        isx=self.isx
        xmin_pml = self.xmin_pml
        xmax_pml = self.xmax_pml
        ymin_pml = self.ymin_pml
        ymax_pml = self.ymax_pml
        zmin_pml = self.zmin_pml
        zmax_pml = self.zmax_pml
        nx=self.gridsize[0]
        ny=self.gridsize[1]
        nz=self.gridsize[2]
            # source term  
        source=Ricker2(nt,dt,0.03,self.f0,1000)
        for i,j,k in ti.ndrange((0,2),(0,2),(0,2)):
            self.sxx[isz+k,isx+i,isy+j,]+=source*(-self.MT[0,0])/8 
            self.syy[isz+k,isx+i,isy+j,]+=source*(-self.MT[1,1])/8 
            self.szz[isz+k,isx+i,isy+j,]+=source*(-self.MT[2,2])/8 

            self.sxy[isz+k,isx+i,isy+j]+=source*(-self.MT[0,1])/8 
            self.sxz[isz+k,isx+i,isy+j]+=source*(-self.MT[0,2])/8 
            self.syz[isz+k,isx+i,isy+j]+=source*(-self.MT[1,2])/8
        # update vx vy vz
        for k,i,j in ti.ndrange((star+1,nz-star),(star,nx-star-1),(star+1,ny-star)):
            x=(i-star)*dx
            y=(j-star)*dy
            z=(k-star)*dz
            dsxxd1=Drd1bm(self.sxx,k,i,j,self.c,star)
            dsxxd2=Drd2bm(self.sxx,k,i,j,self.c,star)
            dsxxd3=Drd3bm(self.sxx,k,i,j,self.c,star)
            dsxxd4=Drd4bm(self.sxx,k,i,j,self.c,star)

            dsxyd1=Drd1bm(self.sxy,k,i,j,self.c,star)
            dsxyd2=Drd2bm(self.sxy,k,i,j,self.c,star)
            dsxyd3=Drd3bm(self.sxy,k,i,j,self.c,star)
            dsxyd4=Drd4bm(self.sxy,k,i,j,self.c,star)

            dsyyd1=Drd1bm(self.syy,k,i,j,self.c,star)
            dsyyd2=Drd2bm(self.syy,k,i,j,self.c,star)
            dsyyd3=Drd3bm(self.syy,k,i,j,self.c,star)
            dsyyd4=Drd4bm(self.syy,k,i,j,self.c,star)

            dsxzd1=Drd1bm(self.sxz,k,i,j,self.c,star)
            dsxzd2=Drd2bm(self.sxz,k,i,j,self.c,star)
            dsxzd3=Drd3bm(self.sxz,k,i,j,self.c,star)
            dsxzd4=Drd4bm(self.sxz,k,i,j,self.c,star)

            dsyzd1=Drd1bm(self.syz,k,i,j,self.c,star)
            dsyzd2=Drd2bm(self.syz,k,i,j,self.c,star)
            dsyzd3=Drd3bm(self.syz,k,i,j,self.c,star)
            dsyzd4=Drd4bm(self.syz,k,i,j,self.c,star)            

            dszzd1=Drd1bm(self.szz,k,i,j,self.c,star)
            dszzd2=Drd2bm(self.szz,k,i,j,self.c,star)
            dszzd3=Drd3bm(self.szz,k,i,j,self.c,star)
            dszzd4=Drd4bm(self.szz,k,i,j,self.c,star)

            dsxxdx =(dsxxd1+dsxxd2+dsxxd3+dsxxd4)/(4*dx)
            dsxydy =(dsxyd1+dsxyd2-dsxyd3-dsxyd4)/(4*dy)
            dsxzdz =(dsxzd1-dsxzd2+dsxzd3-dsxzd4)/(4*dz)
            
            dsxydx =(dsxyd1+dsxyd2+dsxyd3+dsxyd4)/(4*dx)
            dsyydy =(dsyyd1+dsyyd2-dsyyd3-dsyyd4)/(4*dy)
            dsyzdz =(dsyzd1-dsyzd2+dsyzd3-dsyzd4)/(4*dz)
            
            dsxzdx =(dsxzd1+dsxzd2+dsxzd3+dsxzd4)/(4*dx)
            dsyzdy =(dsyzd1+dsyzd2-dsyzd3-dsyzd4)/(4*dy)
            dszzdz =(dszzd1-dszzd2+dszzd3-dszzd4)/(4*dz)

            rho=(self.rho[k,i,j]+self.rho[k,i+1,j]+self.rho[i+1,j+1,k]+self.rho[i,j+1,k]
                  +self.rho[i,j,k+1]+self.rho[i+1,j,k+1]+self.rho[i+1,j+1,k+1]+self.rho[i,j+1,k+1])/8          

            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):

                if(x<=self.xmin_EAL or x>=self.xmax_EAL or y<=self.ymin_EAL or y>=self.ymax_EAL or z<=self.zmin_EAL or z>=self.zmax_EAL):
                    damp=ti.sqrt(self.pml_x[i]**2+self.pml_y[j]**2+self.pml_z[k]**2)
                    kamp=ti.sqrt((self.k_x[i]-1)**2+(self.k_y[j]-1)**2+(self.k_z[k]-1)**2)+1
                    pmln=(kamp-0.5*dt*damp)              
                    pmld=(kamp+0.5*dt*damp)   
                    self.vx[k,i,j]=(pmln*self.vx[k,i,j]+(dsxxdx+dsxydy+dsxzdz)*dt/ rho)/pmld        
                    self.vy[k,i,j]=(pmln*self.vy[k,i,j]+(dsxydx+dsyydy+dsyzdz)*dt/ rho)/pmld
                    self.vz[k,i,j]=(pmln*self.vz[k,i,j]+(dsxzdx+dsyzdy+dszzdz)*dt/rho)/pmld
                else:
                    self.memory_sxx_dx[k,i,j] = self.b_x[i] * self.memory_sxx_dx[k,i,j] + self.a_x[i] * dsxxdx
                    self.memory_sxy_dy[k,i,j] = self.b_y[j] * self.memory_sxy_dy[k,i,j] + self.a_y[j] * dsxydy
                    self.memory_sxz_dz[k,i,j] = self.b_z[k] * self.memory_sxz_dz[k,i,j] + self.a_z[k] * dsxzdz

                    self.memory_sxy_dx[k,i,j] = self.b_x[i] * self.memory_sxy_dx[k,i,j] + self.a_x[i] * dsxydx
                    self.memory_syy_dy[k,i,j] = self.b_y[j] * self.memory_syy_dy[k,i,j] + self.a_y[j] * dsyydy
                    self.memory_syz_dz[k,i,j] = self.b_z[k] * self.memory_syz_dz[k,i,j] + self.a_z[k] * dsyzdz

                    self.memory_sxz_dx[k,i,j] = self.b_x[i] * self.memory_sxz_dx[k,i,j] + self.a_x[i] * dsxzdx
                    self.memory_syz_dy[k,i,j] = self.b_y[j] * self.memory_syz_dy[k,i,j] + self.a_y[j] * dsyzdy
                    self.memory_szz_dz[k,i,j] = self.b_z[k] * self.memory_szz_dz[k,i,j] + self.a_z[k] * dszzdz
                    dsxxdx = dsxxdx/self.k_x[i] + self.memory_sxx_dx[k,i,j]
                    dsxydy = dsxydy/self.k_y[j] + self.memory_sxy_dy[k,i,j]
                    dsxzdz = dsxzdz/self.k_z[k] + self.memory_sxz_dz[k,i,j]

                    dsxydx = dsxydx/self.k_x[i] + self.memory_sxy_dx[k,i,j]
                    dsyydy = dsyydy/self.k_y[j] + self.memory_syy_dy[k,i,j]
                    dsyzdz = dsyzdz/self.k_z[k] + self.memory_syz_dz[k,i,j]

                    dsxzdx = dsxzdx/self.k_x[i] + self.memory_sxz_dx[k,i,j]
                    dsyzdy = dsyzdy/self.k_y[j] + self.memory_syz_dy[k,i,j]
                    dszzdz = dszzdz/self.k_z[k] + self.memory_szz_dz[k,i,j] 
                    self.vx[k,i,j]+=(dsxxdx+dsxydy+dsxzdz)*dt/ rho
                    self.vy[k,i,j]+=(dsxydx+dsyydy+dsyzdz)*dt/ rho 
                    self.vz[k,i,j]+=(dsxzdx+dsyzdy+dszzdz)*dt/ rho 

            else:
                self.vx[k,i,j]+=(dsxxdx+dsxydy+dsxzdz)*dt/ rho
                self.vy[k,i,j]+=(dsxydx+dsyydy+dsyzdz)*dt/ rho 
                self.vz[k,i,j]+=(dsxzdx+dsyzdy+dszzdz)*dt/ rho      
    # implement Dirichlet boundary conditions on the six edges of the grid
        # xmin
        '''
        for i,j,k in ti.ndrange(star,ny,nz):
            self.vx[i,j,k]=0
            self.vy[i,j,k]=0
            self.vz[i,j,k]=0
        # xmax
        for i,j,k in ti.ndrange((nx-star,nx),nx,nz):
            self.vx[i,j,k]=0
            self.vy[i,j,k]=0
            self.vz[i,j,k]=0
        # ymin
        for i,j,k in ti.ndrange(nx,star,nz):
            self.vx[i,j,k]=0
            self.vy[i,j,k]=0 
            self.vz[i,j,k]=0
        # ymax
        for i,j,k in ti.ndrange(nx,(ny-star,ny),ny):
            self.vx[i,j,k]=0
            self.vy[i,j,k]=0
            self.vz[i,j,k]=0  
        # zmin
        for i,j,k in ti.ndrange(nx,ny,star):
            self.vx[i,j,k]=0
            self.vy[i,j,k]=0
            self.vz[i,j,k]=0
        # zmax
        for i,j,k in ti.ndrange(nx,ny,(nz-star,nz)):
            self.vx[i,j,k]=0
            self.vy[i,j,k]=0  
            self.vz[i,j,k]=0  '
        '''
        


       # update sxx szz,syy
        for k,i,j in ti.ndrange((star+1,nz-star),(star+1,nx-star),(star+1,ny-star)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy+dy/2
            z=(k-star)*dz+dz/2
            lam=self.lam[i,j,k]  
            mu =self.mu[i,j,k]
            lam_plus_2mu=lam+2*mu
            dvxd1=Drd1fm(self.vx,k,i,j,self.c,star)
            dvxd2=Drd2fm(self.vx,k,i,j,self.c,star)
            dvxd3=Drd3fm(self.vx,k,i,j,self.c,star)
            dvxd4=Drd4fm(self.vx,k,i,j,self.c,star)

            dvyd1=Drd1fm(self.vy,k,i,j,self.c,star)
            dvyd2=Drd2fm(self.vy,k,i,j,self.c,star)
            dvyd3=Drd3fm(self.vy,k,i,j,self.c,star)
            dvyd4=Drd4fm(self.vy,k,i,j,self.c,star)

            dvzd1=Drd1fm(self.vz,k,i,j,self.c,star)
            dvzd2=Drd2fm(self.vz,k,i,j,self.c,star)
            dvzd3=Drd3fm(self.vz,k,i,j,self.c,star)
            dvzd4=Drd4fm(self.vz,k,i,j,self.c,star)

            dvxdx=(dvxd1+dvxd2+dvxd3+dvxd4) /(4*dx)
            dvxdy=(dvxd1+dvxd2-dvxd3-dvxd4) /(4*dy)
            dvxdz=(dvxd1-dvxd2+dvxd3-dvxd4) /(4*dz)

            dvydx=(dvyd1+dvyd2+dvyd3+dvyd4) /(4*dx)
            dvydy=(dvyd1+dvyd2-dvyd3-dvyd4) /(4*dy)
            dvydz=(dvyd1-dvyd2+dvyd3-dvyd4) /(4*dz) 

            dvzdx=(dvzd1+dvzd2+dvzd3+dvzd4) /(4*dx)
            dvzdy=(dvzd1+dvzd2-dvzd3-dvzd4) /(4*dy)
            dvzdz=(dvzd1-dvzd2+dvzd3-dvzd4) /(4*dz)
        
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                if(x<=self.xmin_EAL or x>=self.xmax_EAL or y<=self.ymin_EAL or y>=self.ymax_EAL or z<=self.zmin_EAL or z>=self.zmax_EAL):
                    damp=ti.sqrt(self.pml_x_half[i]**2+self.pml_y_half[j]**2+self.pml_z_half[k]**2)
                    kamp=ti.sqrt((self.k_x_half[i]-1)**2+(self.k_y_half[j]-1)**2+(self.k_z_half[k]-1)**2)+1
                    pmln=(kamp-0.5*dt*damp)   
                    pmld=(kamp+0.5*dt*damp)
                    self.sxx[k,i,j]=(pmln*self.sxx[k,i,j]+(lam_plus_2mu*dvxdx+lam*(dvzdz+dvydy))*dt)/pmld        
                    self.syy[k,i,j]=(pmln*self.syy[k,i,j]+(lam_plus_2mu*dvydy+lam*(dvxdx+dvzdz))*dt)/pmld     
                    self.szz[k,i,j]=(pmln*self.szz[k,i,j]+(lam_plus_2mu*dvzdz+lam*(dvxdx+dvydy))*dt )/pmld

                        

                    self.sxy[k,i,j]=(pmln*self.sxy[k,i,j]+mu*(dvxdy+dvydx)*dt)/pmld
                    self.sxz[k,i,j]=(pmln*self.sxz[k,i,j]+mu*(dvxdz+dvzdx)*dt )/pmld
                    self.syz[k,i,j]=(pmln*self.syz[k,i,j]+ mu*(dvydz+dvzdy)*dt)/pmld
                else:
            
                    self.memory_dvx_dx[k,i,j] = self.b_x_half[i] * self.memory_dvx_dx[k,i,j] + self.a_x_half[i]*dvxdx
                    self.memory_dvy_dy[k,i,j] = self.b_y_half[j] * self.memory_dvy_dy[k,i,j] + self.a_y_half[j]*dvydy
                    self.memory_dvz_dz[k,i,j] = self.b_z_half[k] * self.memory_dvz_dz[k,i,j] + self.a_z_half[k]*dvzdz

                    self.memory_dvy_dx[k,i,j] = self.b_x_half[i] * self.memory_dvy_dx[k,i,j] + self.a_x_half[i]*dvydx 
                    self.memory_dvx_dy[k,i,j] = self.b_y_half[j] * self.memory_dvx_dy[k,i,j] + self.a_y_half[j]*dvxdy 

                    self.memory_dvz_dx[k,i,j] = self.b_x_half[i] * self.memory_dvz_dx[k,i,j] + self.a_x_half[i]*dvzdx 
                    self.memory_dvx_dz[k,i,j] = self.b_z_half[k] * self.memory_dvx_dz[k,i,j] + self.a_z_half[k]*dvxdz

                    self.memory_dvz_dy[k,i,j] = self.b_y_half[j] * self.memory_dvz_dy[k,i,j] + self.a_y_half[j]*dvzdy 
                    self.memory_dvy_dz[k,i,j] = self.b_z_half[k] * self.memory_dvy_dz[k,i,j] + self.a_z_half[k]*dvydz 

                    dvxdx = dvxdx /self.k_x_half[i] + self.memory_dvx_dx[k,i,j]
                    dvydy = dvydy /self.k_y_half[j] + self.memory_dvy_dy[k,i,j]
                    dvzdz = dvzdz /self.k_z_half[k] + self.memory_dvz_dz[k,i,j]

                    dvydx = dvydx /self.k_x_half[i] + self.memory_dvy_dx[k,i,j]
                    dvxdy = dvxdy /self.k_y_half[j] + self.memory_dvx_dy[k,i,j]

                    dvzdx = dvzdx /self.k_x_half[i] + self.memory_dvz_dx[k,i,j]
                    dvxdz = dvxdz /self.k_z_half[k] + self.memory_dvx_dz[k,i,j]

                    dvzdy = dvzdy /self.k_y_half[j] + self.memory_dvz_dy[k,i,j]
                    dvydz = dvydz /self.k_z_half[k] + self.memory_dvy_dz[k,i,j]
                    self.sxx[k,i,j]+=(lam_plus_2mu*dvxdx+lam*(dvzdz+dvydy))*dt
                    self.syy[k,i,j]+=(lam_plus_2mu*dvydy+lam*(dvxdx+dvzdz))*dt
                    self.szz[k,i,j]+=(lam_plus_2mu*dvzdz+lam*(dvxdx+dvydy))*dt 
                    self.sxy[k,i,j]+= mu*(dvxdy+dvydx)*dt 
                    self.sxz[k,i,j]+= mu*(dvxdz+dvzdx)*dt 
                    self.syz[k,i,j]+= mu*(dvydz+dvzdy)*dt 
            else:
                self.sxx[k,i,j]+=(lam_plus_2mu*dvxdx+lam*(dvzdz+dvydy))*dt
                self.syy[k,i,j]+=(lam_plus_2mu*dvydy+lam*(dvxdx+dvzdz))*dt
                self.szz[k,i,j]+=(lam_plus_2mu*dvzdz+lam*(dvxdx+dvydy))*dt 
                self.sxy[k,i,j]+= mu*(dvxdy+dvydx)*dt 
                self.sxz[k,i,j]+= mu*(dvxdz+dvzdx)*dt 
                self.syz[k,i,j]+= mu*(dvydz+dvzdy)*dt
        for i in range(self.datasize):
            self.data[nt,i]=self.vx[self.rsz[0,i],self.rsx[0,i],self.rsy[0,i]] 

        
   

    @staticmethod
    def diff_coff(order:int):
        b=np.zeros((order))
        b[0]=1
        A=np.zeros((order,order))
        for i in range(order):
            for j in range(order):
                A[i,j]=(2*j+1)**(2*i+1)
        c_np=np.linalg.solve(A,b)  # Calculate the finite difference coefficients
        c=ti.field(dtype=ti.f32,shape=(order,))
        c.from_numpy(c_np)
        return c
    
    def SetADEPML3D(self,pml_surface,parameter:dict):
        vp_max=parameter["vp_max"]  
        dx=self.dx
        dy=self.dy
        dz=self.dz
        dt=self.dt
        nx=self.gridsize[0]
        ny=self.gridsize[1]
        nz=self.gridsize[2]
        xmin=self.xmin
        ymin=self.ymin
        xmax=self.xmax
        ymax=self.ymax
        zmin=self.zmin
        zmax=self.zmax
        pml_x_thick=parameter["pml_x_thick"]
        pml_y_thick=parameter["pml_y_thick"]  
        pml_z_thick=parameter["pml_z_thick"]
        Rcoef =parameter["Rcoef"]
        alpha_max_pml=parameter["alpha_max_pml"]
        k_max_pml=parameter["kmax_pml"]
        theta=parameter["theta"]
        star=self.star
        d0_x = -3.0 * vp_max * ti.log(Rcoef) / (2.0 * pml_x_thick)
        d0_y = -3.0 * vp_max * ti.log(Rcoef) / (2.0 * pml_y_thick)
        d0_z = -3.0 * vp_max * ti.log(Rcoef) / (2.0 * pml_z_thick)
        # set pml boudary
        Use_PML_X_Left  = pml_surface[0] 
        Use_PML_X_Right = pml_surface[1]
        Use_PML_Y_Left  = pml_surface[2] 
        Use_PML_Y_Right = pml_surface[3]
        Use_PML_Z_Up    = pml_surface[4]
        Use_PML_Z_Bottom= pml_surface[5]
        # The location of pml boundary
        self.xmin_pml = xmin + pml_x_thick
        self.xmax_pml = xmax - pml_x_thick
        self.ymin_pml = ymin + pml_y_thick
        self.ymax_pml = ymax - pml_y_thick
        self.zmin_pml = zmin + pml_z_thick
        self.zmax_pml = zmax - pml_z_thick
        # x direction 
        # define damping profile at the grid point  
        for i in range(nx):
            x=(i-star)*dx
            if (x<= self.xmin_pml and Use_PML_X_Left) :
                abscissa_normalized=(self.xmin_pml-x) / pml_x_thick
                pml_dx_temp  = d0_x * abscissa_normalized**2
                alpha_x_temp= alpha_max_pml * (1- abscissa_normalized)
                k_x_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            elif (x>=self.xmax_pml and Use_PML_X_Right ):
                abscissa_normalized=(x-self.xmax_pml) / pml_x_thick
                pml_dx_temp  = d0_x * abscissa_normalized**2
                alpha_x_temp= alpha_max_pml * (1- abscissa_normalized)
                k_x_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            else:
                pml_dx_temp = 0
                alpha_x_temp=0
                k_x_temp=1
            self.pml_x[i]= pml_dx_temp
            self.alpha_x[i]=alpha_x_temp
            self.k_x[i]=k_x_temp
            if self.alpha_x[i]<0:
                self.alpha_x[i]=0
        #  define damping profile at the half grid points
        for i in range(nx):
            x_half=(i-star)*dx+dx/2
            if (x_half<= self.xmin_pml and Use_PML_X_Left) :
                abscissa_normalized=(self.xmin_pml-x_half) / pml_x_thick
                pml_half_x_temp  = d0_x * abscissa_normalized**2
                alpha_half_x_temp= alpha_max_pml * (1- abscissa_normalized)
                k_half_x_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            elif (x_half>=self.xmax_pml and Use_PML_X_Right ):
                abscissa_normalized=(x_half-self.xmax_pml) / pml_x_thick
                pml_half_x_temp  = d0_x * abscissa_normalized**2
                alpha_half_x_temp= alpha_max_pml * (1- abscissa_normalized)
                k_half_x_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            else:
                pml_half_x_temp  = 0
                alpha_half_x_temp=0
                k_half_x_temp=1
            self.pml_x_half[i]= pml_half_x_temp
            self.alpha_x_half[i]=alpha_half_x_temp
            self.k_x_half[i]=k_half_x_temp
            if self.alpha_x_half[i]<0:
                self.alpha_x_half[i]=0
        # y direction 
        # define damping profile at the grid points
        for i in range(ny):
            y=(i-star)*dy
            if (y<= self.ymin_pml and Use_PML_Y_Left) :
                abscissa_normalized=(self.ymin_pml-y) / pml_y_thick
                pml_y_temp  = d0_y * abscissa_normalized**2
                alpha_y_temp= alpha_max_pml * (1- abscissa_normalized)
                k_y_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            elif (y>=self.ymax_pml and Use_PML_Y_Right ):
                abscissa_normalized=(y-self.ymax_pml) / pml_y_thick
                pml_y_temp  = d0_y * abscissa_normalized**2
                alpha_y_temp= alpha_max_pml * (1- abscissa_normalized)
                k_y_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            else:
                pml_y_temp  = 0
                alpha_y_temp=0
                k_y_temp=1
            self.pml_y[i]= pml_y_temp
            self.alpha_y[i]=alpha_y_temp
            self.k_y[i]=k_y_temp
            if self.alpha_y[i]<0:
                self.alpha_y[i]=0
        #  define damping profile at the half grid points
        for i in range(ny):
            y_half=(i-star)*dy+dy/2
            if (y_half<= self.ymin_pml and Use_PML_Y_Left) :
                abscissa_normalized=(self.ymin_pml-y_half) / pml_y_thick
                pml_half_y_temp  = d0_y * abscissa_normalized**2
                alpha_half_y_temp= alpha_max_pml * (1- abscissa_normalized)
                k_half_y_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            elif (y_half>=self.ymax_pml and Use_PML_Y_Right ):
                abscissa_normalized=(y_half-self.ymax_pml) / pml_y_thick
                pml_half_y_temp  = d0_y * abscissa_normalized**2
                alpha_half_y_temp= alpha_max_pml * (1- abscissa_normalized)
                k_half_y_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            else:
                pml_half_y_temp  = 0
                alpha_half_y_temp=0
                k_half_y_temp=1
            self.pml_y_half[i]= pml_half_y_temp
            self.alpha_y_half[i]=alpha_half_y_temp
            self.k_y_half[i]=k_half_y_temp
            if self.alpha_y_half[i]<0:
                self.alpha_y_half[i]=0
        # z direction 
        # define damping profile at the grid points
        for i in range(nz):
            z=(i-star)*dz
            if (z<= self.zmin_pml and Use_PML_Z_Up) :
                abscissa_normalized=(self.zmin_pml-z) / pml_z_thick
                pml_z_temp  = d0_z * abscissa_normalized**2
                alpha_z_temp= alpha_max_pml * (1- abscissa_normalized)
                k_z_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            elif (z>=self.zmax_pml and Use_PML_Z_Bottom ):
                abscissa_normalized=(z-self.zmax_pml) / pml_z_thick
                pml_z_temp  = d0_z * abscissa_normalized**2
                alpha_z_temp= alpha_max_pml * (1- abscissa_normalized)
                k_z_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            else:
                pml_z_temp  = 0
                alpha_z_temp=0
                k_z_temp=1
            self.pml_z[i]= pml_z_temp
            self.alpha_z[i]=alpha_z_temp
            self.k_z[i]=k_z_temp
            if self.alpha_z[i]<0:
                self.alpha_z[i]=0
        #  define damping profile at the half grid points
        for i in range(nz):
            z_half=(i-star)*dz+dz/2
            if (z_half<= self.zmin_pml and Use_PML_Z_Up) :
                abscissa_normalized=(self.zmin_pml-z_half) / pml_z_thick
                pml_half_z_temp  = d0_z * abscissa_normalized**2
                alpha_half_z_temp= alpha_max_pml * (1- abscissa_normalized)
                k_half_z_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            elif (z_half>=self.zmax_pml and Use_PML_Z_Bottom ):
                abscissa_normalized=(z_half-self.zmax_pml) / pml_z_thick
                pml_half_z_temp  = d0_z * abscissa_normalized**2
                alpha_half_z_temp= alpha_max_pml * (1- abscissa_normalized)
                k_half_z_temp = 1 + (k_max_pml - 1) * abscissa_normalized**2
            else:
                pml_half_z_temp  = 0
                alpha_half_z_temp=0
                k_half_z_temp=1
            self.pml_z_half[i]= pml_half_z_temp
            self.alpha_z_half[i]=alpha_half_z_temp
            self.k_z_half[i]=k_half_z_temp
            if self.alpha_z_half[i]<0:
                self.alpha_z_half[i]=0

        # PML damping parameters for time steps of memory variable 
        for i in range(nx):
            self.b_x[i] =(1-(1-theta)*dt*(self.pml_x[i]/self.k_x[i] + self.alpha_x[i]))/(1+theta*dt*(self.pml_x[i]/self.k_x[i] + self.alpha_x[i]))
            self.b_x_half[i] = (1-(1-theta)*dt*(self.pml_x_half[i]/self.k_x_half[i] +self.alpha_x_half[i]))/(1+theta*dt*(self.pml_x_half[i]/self.k_x_half[i]+ self.alpha_x_half[i]))
            if np.abs(self.pml_x[i])>1e-6:
                self.a_x[i] = - dt*self.pml_x[i]/(self.k_x[i]* self.k_x[i])/(1+theta*dt*(self.pml_x[i]/self.k_x[i] + self.alpha_x[i]))
            if np.abs(self.pml_x_half[i])>1e-6:
                self.a_x_half[i] = - dt*self.pml_x_half[i]/(self.k_x_half[i]* self.k_x_half[i])/(1+theta*dt*(self.pml_x_half[i]/self.k_x_half[i] + self.alpha_x_half[i]))           
        for i in range(ny):
            self.b_y[i] = (1-(1-theta)*dt*(self.pml_y[i]/self.k_y[i] + self.alpha_y[i]))/(1+theta*dt*(self.pml_y[i]/self.k_y[i] + self.alpha_y[i]))
            self.b_y_half[i] = (1-(1-theta)*dt*(self.pml_y_half[i]/self.k_y_half[i] + self.alpha_y_half[i]))/(1+theta*dt*(self.pml_y_half[i]/self.k_y_half[i] + self.alpha_y_half[i]))
            if np.abs(self.pml_y[i]) > 1e-6:
                self.a_y[i] = -dt * self.pml_y[i] / (self.k_y[i] * self.k_y[i]) / (1 + theta * dt * (self.pml_y[i] / self.k_y[i] + self.alpha_y[i]))
            if np.abs(self.pml_y_half[i]) > 1e-6:
                self.a_y_half[i] = -dt * self.pml_y_half[i] / (self.k_y_half[i] * self.k_y_half[i]) / (1 + theta * dt * (self.pml_y_half[i] / self.k_y_half[i] + self.alpha_y_half[i]))
        
        for i in range(nz):
            self.b_z[i]      = (1-(1-theta)*dt*(     self.pml_z[i]/self.k_z[i] + self.alpha_z[i]))/(1+theta*dt*(self.pml_z[i]/self.k_z[i] + self.alpha_z[i]))
            self.b_z_half[i] = (1-(1-theta)*dt*(self.pml_z_half[i]/self.k_z_half[i] +self.alpha_z_half[i]))/(1+theta*dt*(self.pml_z_half[i]/self.k_z_half[i]+ self.alpha_z_half[i])) 
            if np.abs(self.pml_z[i])>1e-6:
                self.a_z[i] = - dt*self.pml_z[i]/(self.k_z[i]* self.k_z[i])/(1+theta*dt*(self.pml_z[i]/self.k_z[i] + self.alpha_z[i]))
            if np.abs(self.pml_z_half[i])>1e-6:
                self.a_z_half[i] = - dt*self.pml_z_half[i]/(self.k_z_half[i]* self.k_z_half[i])/(1+theta*dt*(self.pml_z_half[i]/self.k_z_half[i] + self.alpha_z_half[i]))   


    




    

