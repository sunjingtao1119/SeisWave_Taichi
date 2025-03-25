import taichi as ti
import numpy as np
from src.Differential3D import Dx3bm,Dx3fm,Dy3bm,Dy3fm,Dz3bm,Dz3fm
from src.BaseFun import Ricker
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
                 nt:int,               
                 accuracy:int,
                 freq=100.,
                 src_scale=1.,
                 fieldtype=ti.f32
                ):
        # Initialize model parameters
        self.vs=vs
        self.vp=vp
        self.rho=rho
        self.gridsize=vs.shape
        self.dx=dx
        self.dy=dy
        self.dz=dz
        self.dl=ti.sqrt(dx**2+dy**2+dz**2)
        self.xmin=0
        self.ymin=0
        self.zmin=0
        self.star=accuracy
        self.c=self.diff_coff(accuracy)
        self.xmax=self.xmin+dx*(self.gridsize[0]-1-2*accuracy)    # Calculate the maximum x-coordinate based on grid size and spacing
        self.ymax=self.zmin+dy*(self.gridsize[1]-1-2*accuracy)    # Calculate the maximum z-coordinate based on grid size and spacing
        self.zmax=self.zmin+dz*(self.gridsize[2]-1-2*accuracy)
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
        self.src_scale=src_scale
        self.data=ti.field(dtype=ti.f32,shape=nt)
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
        self.pml_x       =ti.field(fieldtype,shape=self.gridsize[0])
        self.pml_x_half  =ti.field(fieldtype,shape=self.gridsize[0])   # half grid
        self.alpha_x     =ti.field(fieldtype,shape=self.gridsize[0])
        self.alpha_x_half=ti.field(fieldtype,shape=self.gridsize[0])   # half grid
        self.k_x         =ti.field(fieldtype,shape=self.gridsize[0])
        self.k_x_half    =ti.field(fieldtype,shape=self.gridsize[0])   # half grid
        self.b_x         =ti.field(fieldtype,shape=self.gridsize[0])
        self.b_x_half    =ti.field(fieldtype,shape=self.gridsize[0])   # half grid
        self.a_x         =ti.field(fieldtype,shape=self.gridsize[0])
        self.a_x_half    =ti.field(fieldtype,shape=self.gridsize[0])   # half grid

        self.pml_y       =ti.field(fieldtype,shape=self.gridsize[1])
        self.pml_y_half  =ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.alpha_y     =ti.field(fieldtype,shape=self.gridsize[1])
        self.alpha_y_half=ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.k_y         =ti.field(fieldtype,shape=self.gridsize[1])
        self.k_y_half    =ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.b_y         =ti.field(fieldtype,shape=self.gridsize[1])
        self.b_y_half    =ti.field(fieldtype,shape=self.gridsize[1])   # half grid
        self.a_y         =ti.field(fieldtype,shape=self.gridsize[1])
        self.a_y_half    =ti.field(fieldtype,shape=self.gridsize[1])   # half grid


        self.pml_z       =ti.field(fieldtype,shape=self.gridsize[2])
        self.pml_z_half  =ti.field(fieldtype,shape=self.gridsize[2])
        self.alpha_z     =ti.field(fieldtype,shape=self.gridsize[2])
        self.alpha_z_half=ti.field(fieldtype,shape=self.gridsize[2])
        self.k_z         =ti.field(fieldtype,shape=self.gridsize[2])
        self.k_z_half    =ti.field(fieldtype,shape=self.gridsize[2])
        self.b_z         =ti.field(fieldtype,shape=self.gridsize[2])
        self.b_z_half    =ti.field(fieldtype,shape=self.gridsize[2])
        self.a_z         =ti.field(fieldtype,shape=self.gridsize[2])
        self.a_z_half    =ti.field(fieldtype,shape=self.gridsize[2])
        self.xmin_pml=self.xmin
        self.xmax_pml=self.xmax
        self.zmin_pml=self.zmin
        self.zmax_pml=self.zmax
        # Initialize memory variables for PML (Perfectly Matched Layer)
        self.vx_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vx_y=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vx_z=ti.field(dtype=fieldtype,shape=self.gridsize)
      
        self.vy_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vy_y=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vy_z=ti.field(dtype=fieldtype,shape=self.gridsize)

        self.vz_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vz_y=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vz_z=ti.field(dtype=fieldtype,shape=self.gridsize)

        self.sxx_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxx_y=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxx_z=ti.field(dtype=fieldtype,shape=self.gridsize)

        self.syy_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.syy_y=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.syy_z=ti.field(dtype=fieldtype,shape=self.gridsize)

        self.sxy_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxy_y=ti.field(dtype=fieldtype,shape=self.gridsize)

        self.sxz_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxz_z=ti.field(dtype=fieldtype,shape=self.gridsize)

        self.syz_y=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.syz_z=ti.field(dtype=fieldtype,shape=self.gridsize)

        self.szz_x=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.szz_y=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.szz_z=ti.field(dtype=fieldtype,shape=self.gridsize)
        
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
    def update_SSG_SPML(self,nt:int):
        star=self.star
        dx=self.dx
        dy=self.dy
        dz=self.dz
        dt=self.dt
        isz=self.isz
        isy=self.isy
        isx=self.isx
        src_scale=self.src_scale
        # Boundary position of PML
        xmin_pml = self.xmin_pml
        xmax_pml = self.xmax_pml
        ymin_pml = self.ymin_pml
        ymax_pml = self.ymax_pml
        zmin_pml = self.zmin_pml
        zmax_pml = self.zmax_pml
        nx=self.gridsize[0]
        ny=self.gridsize[1]
        nz=self.gridsize[2]
        # update vx
        for i,j,k in ti.ndrange((star,nx-star-1),(star+1,ny-star),(star+1,nz-star)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy
            z=(k-star)*dz
            dsxxdx=Dx3fm(self.sxx,i,j,k,self.c,star)/dx
            dsxydy=Dy3bm(self.sxy,i,j,k,self.c,star)/dy
            dsxzdz=Dz3bm(self.sxz,i,j,k,self.c,star)/dz
            rho=self.rho[i,j,k]
            rho_over=1/rho
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                pmlxn=(1-0.5*dt*self.pml_x_half[i])
                pmlyn=(1-0.5*dt*self.pml_y[j])
                pmlzn=(1-0.5*dt*self.pml_z[k])

                pmlxd=(1+0.5*dt*self.pml_x_half[i])
                pmlyd=(1+0.5*dt*self.pml_y[j])
                pmlzd=(1+0.5*dt*self.pml_z[k])

                self.vx_x[i,j,k]=(pmlxn*self.vx_x[i,j,k]+dsxxdx*dt*rho_over)/pmlxd
                self.vx_y[i,j,k]=(pmlyn*self.vx_y[i,j,k]+dsxydy*dt*rho_over)/pmlyd
                self.vx_z[i,j,k]=(pmlzn*self.vx_z[i,j,k]+dsxzdz*dt*rho_over)/pmlzd
                self.vx[i,j,k]=self.vx_x[i,j,k]+self.vx_y[i,j,k]+self.vx_z[i,j,k]
            else:
                self.vx[i,j,k] +=(dsxxdx+dsxydy+dsxzdz)*dt*rho_over     
            
            #self.vx[i,j,k]+=(dsxxdx+dsxydy+dsxzdz)*dt*rho_over   
        # update vy
        for i, j, k in ti.ndrange((star+1, nx - star ), (star , ny - star-1), (star+1, nz - star )):
            x = (i-star)*dx
            y = (j-star)*dy+dy/2
            z = (k-star)*dz
            rho_half_y = 0.5*(self.rho[i,j,k]+self.rho[i,j + 1,k])
            rho_over=1/rho_half_y
            dsxydx = Dx3bm(self.sxy, i, j, k, self.c, star) / dx
            dsyydy = Dy3fm(self.syy, i, j, k, self.c, star) / dy
            dsyzdz = Dz3bm(self.syz, i, j, k, self.c, star) / dz
            if (x <= xmin_pml or x >= xmax_pml or y <= ymin_pml or y >= ymax_pml or z <= zmin_pml or z >= zmax_pml):
                pmlxn=(1-0.5*dt*self.pml_x[i])
                pmlyn=(1-0.5*dt*self.pml_y_half[j])
                pmlzn=(1-0.5*dt*self.pml_z[k])

                pmlxd=(1+0.5*dt*self.pml_x[i])
                pmlyd=(1+0.5*dt*self.pml_y_half[j])
                pmlzd=(1+0.5*dt*self.pml_z[k])

                self.vy_x[i,j,k]=(pmlxn*self.vy_x[i,j,k]+dt*dsxydx*rho_over)/pmlxd
                self.vy_y[i,j,k]=(pmlyn*self.vy_y[i,j,k]+dt*dsyydy*rho_over)/pmlyd
                self.vy_z[i,j,k]=(pmlzn*self.vy_z[i,j,k]+dt*dsyzdz*rho_over)/pmlzd
                self.vy[i,j,k] = self.vy_x[i,j,k]+self.vy_y[i,j,k]+self.vy_z[i,j,k]
                
            else:              
                self.vy[i,j,k] +=(dsxydx+dsyydy+dsyzdz)*dt*rho_over 
        #update vz
        for i,j,k in ti.ndrange((star+1,nx-star),(star+1,ny-star),(star,nz-star-1)):
            x=(i-star)*dx
            y=(j-star)*dy
            z=(k-star)*dz+dz/2  
            rho_half_x_half_z=0.25*(self.rho[i,j,k]+self.rho[i,j+1,k]+self.rho[i+1,j+1,k]+self.rho[i+1,j,k])
            rho_over=1/rho_half_x_half_z
            dsxzdx=Dx3bm(self.sxz,i,j,k,self.c,star)/dx
            dsyzdy=Dy3bm(self.syz,i,j,k,self.c,star)/dy
            dszzdz=Dz3fm(self.szz,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                pmlxn=(1-0.5*dt*self.pml_x[i])
                pmlyn=(1-0.5*dt*self.pml_y[j])
                pmlzn=(1-0.5*dt*self.pml_z_half[k])

                pmlxd=(1+0.5*dt*self.pml_x[i])
                pmlyd=(1+0.5*dt*self.pml_y[j])
                pmlzd=(1+0.5*dt*self.pml_z_half[k])

                self.vz_x[i,j,k]=(pmlxn*self.vz_x[i,j,k]+dt*dsxzdx*rho_over)/pmlxd
                self.vz_y[i,j,k]=(pmlyn*self.vz_y[i,j,k]+dt*dsyzdy*rho_over)/pmlyd
                self.vz_z[i,j,k]=(pmlzn*self.vz_z[i,j,k]+dt*dszzdz*rho_over)/pmlzd
                self.vz[i,j,k]=self.vz_x[i,j,k]+self.vz_y[i,j,k]+self.vz_z[i,j,k]
                
            else:
                self.vz[i,j,k]+=(dsxzdx+dsyzdy+dszzdz)*dt*rho_over
    # implement Dirichlet boundary conditions on the six edges of the grid
        '''
        # xmin
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
       # source term  
        self.vz[isx,isy,isz]+=dt*Ricker(nt,dt,self.f0,src_scale)/self.rho[isx,isy,isz] 
       # update sxx szz,syy
        for i,j,k in ti.ndrange((star+1,nx-star),(star+1,ny-star),(star+1,nz-star)):
            x=(i-star)*dx
            y=(j-star)*dy
            z=(k-star)*dz
            lam=self.lam[i,j,k]   
            mu =self.mu[i,j,k]
            lam_plus_2mu=lam+2*mu
            dvxdx=Dx3bm(self.vx,i,j,k,self.c,star)/dx
            dvydy=Dy3bm(self.vy,i,j,k,self.c,star)/dy
            dvzdz=Dz3bm(self.vz,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                pmlxn=(1-0.5*dt*self.pml_x[i])
                pmlyn=(1-0.5*dt*self.pml_y[j])
                pmlzn=(1-0.5*dt*self.pml_z[k])

                pmlxd=(1+0.5*dt*self.pml_x[i])
                pmlyd=(1+0.5*dt*self.pml_y[j])
                pmlzd=(1+0.5*dt*self.pml_z[k])

                self.sxx_x[i,j,k]=(pmlxn*self.sxx_x[i,j,k]+dt*lam_plus_2mu*dvxdx)/pmlxd
                self.sxx_y[i,j,k]=(pmlyn*self.sxx_y[i,j,k]+dt*lam*dvydy)/pmlyd
                self.sxx_z[i,j,k]=(pmlzn*self.sxx_z[i,j,k]+dt*lam*dvzdz)/pmlzd

                self.syy_x[i,j,k]=(pmlxn*self.syy_x[i,j,k]+dt*lam*dvxdx)/pmlxd
                self.syy_y[i,j,k]=(pmlyn*self.syy_y[i,j,k]+dt*lam_plus_2mu*dvydy)/pmlyd
                self.syy_z[i,j,k]=(pmlzn*self.syy_z[i,j,k]+dt*lam*dvzdz)/pmlzd

                self.szz_x[i,j,k]=(pmlxn*self.szz_x[i,j,k]+dt*lam*dvxdx)/pmlxd
                self.szz_y[i,j,k]=(pmlyn*self.szz_y[i,j,k]+dt*lam*dvydy)/pmlyd
                self.szz_z[i,j,k]=(pmlzn*self.szz_z[i,j,k]+dt*lam_plus_2mu*dvzdz)/pmlzd              

                self.sxx[i,j,k]=self.sxx_x[i,j,k]+self.sxx_y[i,j,k]+self.sxx_z[i,j,k]
                self.syy[i,j,k]=self.syy_x[i,j,k]+self.syy_y[i,j,k]+self.syy_z[i,j,k]
                self.szz[i,j,k]=self.szz_x[i,j,k]+self.szz_y[i,j,k]+self.szz_z[i,j,k]
            else:
                self.sxx[i,j,k]+=(lam_plus_2mu*dvxdx+lam*(dvzdz+dvydy))*dt
                self.syy[i,j,k]+=(lam_plus_2mu*dvydy+lam*(dvxdx+dvzdz))*dt
                self.szz[i,j,k]+=(lam_plus_2mu*dvzdz+lam*(dvxdx+dvydy))*dt    
        for i,j,k in ti.ndrange((star+1,nx-star),(star+1,ny-star),(star,nz-star)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy+dy/2
            z=(k-star)*dz
            mu_half_x_half_y=0.25*(self.mu[i,j+1,k]+self.mu[i,j,k]+self.mu[i,j,k]+self.mu[i+1,j,k])
            dvydx = Dx3fm(self.vy,i,j,k,self.c,star)/dx
            dvxdy = Dy3fm(self.vx,i,j,k,self.c,star)/dy   
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml ):
                pmlxn=(1-0.5*dt*self.pml_x_half[i])
                pmlyn=(1-0.5*dt*self.pml_y_half[j])
                pmlxd=(1+0.5*dt*self.pml_x_half[i])
                pmlyd=(1+0.5*dt*self.pml_y_half[j])        
                self.sxy_x[i,j,k]=(pmlxn*self.sxy_x[i,j,k]+dt*mu_half_x_half_y*dvydx)/pmlxd
                self.sxy_y[i,j,k]=(pmlyn*self.sxy_y[i,j,k]+dt*mu_half_x_half_y*dvxdy)/pmlyd
                self.sxy[i,j,k]=self.sxy_x[i,j,k]+self.sxy_y[i,j,k]
            else:
                self.sxy[i,j,k]+=mu_half_x_half_y*(dvxdy+dvydx)*dt 

        for i,j,k in ti.ndrange((star,nx-star-1),(star,ny-star),(star,nz-star-1)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy
            z=(k-star)*dz+dz/2
            mu_half_x_half_z=0.25*(self.mu[i,j,k]+self.mu[i+1,j,k]+self.mu[i,j,k]+self.mu[i,j+1,k])
            dvzdx = Dx3fm(self.vz,i,j,k,self.c,star)/dx
            dvxdz = Dz3fm(self.vx,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml ):
                pmlxn=(1-0.5*dt*self.pml_x_half[i])
                pmlzn=(1-0.5*dt*self.pml_z_half[k])

                pmlxd=(1+0.5*dt*self.pml_x_half[i])
                pmlzd=(1+0.5*dt*self.pml_z_half[k])

                self.sxz_x[i,j,k]=(pmlxn*self.sxz_x[i,j,k]+dt*mu_half_x_half_z*dvzdx)/pmlxd
                self.sxz_z[i,j,k]=(pmlzn*self.sxz_z[i,j,k]+dt*mu_half_x_half_z*dvxdz)/pmlzd

                self.sxz[i,j,k]=self.sxz_x[i,j,k]+self.sxz_z[i,j,k]
            else:
                self.sxz[i,j,k]+=mu_half_x_half_z*(dvxdz+dvzdx)*dt 

        for i,j,k in ti.ndrange((star,nx-star),(star,ny-star-1),(star,nz-star-1)):
            x=(i-star)*dx
            y=(j-star)*dy+dy/2
            z=(k-star)*dz+dz/2
            mu_half_y_half_z=0.25*(self.mu[i,j+1,k]+self.mu[i,j,k]+self.mu[i,j,k+1]+self.mu[i,j,k])
            dvzdy = Dy3fm(self.vz,i,j,k,self.c,star)/dy
            dvydz = Dz3fm(self.vy,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml ):
                pmlyn=(1-0.5*dt*self.pml_y_half[j])
                pmlzn=(1-0.5*dt*self.pml_z_half[k])

                pmlyd=(1+0.5*dt*self.pml_y_half[j])
                pmlzd=(1+0.5*dt*self.pml_z_half[k]) 

                self.syz_y[i,j,k]=(pmlyn*self.syz_y[i,j,k]+ dt*mu_half_y_half_z*dvzdy)/pmlyd
                self.syz_z[i,j,k]=(pmlzn*self.syz_z[i,j,k]+ dt*mu_half_y_half_z*dvydz)/pmlzd
                self.syz[i,j,k]=self.syz_y[i,j,k]+self.syz_z[i,j,k]
            else:
                self.syz[i,j,k]+=mu_half_y_half_z*(dvydz+dvzdy)*dt 
        
        self.data[nt]=self.vx[self.rsx,self.rsy,self.rsz]
    
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


    




    

