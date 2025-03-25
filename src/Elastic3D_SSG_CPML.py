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
        self.k_y        =ti.field(fieldtype,shape=self.gridsize[1])
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
    def update_SSG(self,nt:int):
        star=self.star
        dx=self.dx
        dy=self.dy
        dz=self.dz
        dt=self.dt
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
        # update vx
        for i,j,k in ti.ndrange((star,nx-star-1),(star+1,ny-star),(star+1,nz-star)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy
            z=(k-star)*dz
            dsxx =Dx3fm(self.sxx,i,j,k,self.c,star)/dx
            dsxyy=Dy3bm(self.sxy,i,j,k,self.c,star)/dy
            dsxzz=Dz3bm(self.sxz,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_sxx_dx[i,j,k] = self.b_x_half[i] * self.memory_sxx_dx[i,j,k] + self.a_x_half[i] * dsxx
                self.memory_sxy_dy[i,j,k] = self.b_y[j] * self.memory_sxy_dy[i,j,k] + self.a_y[j] * dsxyy
                self.memory_sxz_dz[i,j,k] = self.b_z[k] * self.memory_sxz_dz[i,j,k] + self.a_z[k] * dsxzz
                dsxx  = dsxx /self.k_x_half[i] + self.memory_sxx_dx[i,j,k]
                dsxyy = dsxyy/self.k_y[j] + self.memory_sxy_dy[i,j,k]
                dsxzz = dsxzz/self.k_z[k] + self.memory_sxz_dz[i,j,k]
            self.vx[i,j,k]+=(dsxx+dsxzz+dsxyy)*dt/self.rho[i,j,k]      
        # update vy
        for i, j, k in ti.ndrange((star+1,nx-star ),(star,ny-star-1),(star+1,nz-star)):
            x = (i-star)*dx
            y = (j-star)*dy+dy/2
            z = (k-star)*dz
            rho_half_y=0.5*(self.rho[i,j,k]+self.rho[i,j+1,k])
            dsxy = Dx3bm(self.sxy,i,j,k,self.c,star)/dx
            dsyy = Dy3fm(self.syy,i,j,k,self.c,star)/dy
            dsyz = Dz3bm(self.syz,i,j,k,self.c,star)/dz
            if (x <= xmin_pml or x >= xmax_pml or y <= ymin_pml or y >= ymax_pml or z <= zmin_pml or z >= zmax_pml):
                self.memory_sxy_dx[i,j,k] = self.b_x[i] * self.memory_sxy_dx[i,j,k] + self.a_x[i] * dsxy
                self.memory_syy_dy[i,j,k] = self.b_y_half[j] * self.memory_syy_dy[i,j,k] + self.a_y_half[j] * dsyy
                self.memory_syz_dz[i,j,k] = self.b_z[k] * self.memory_syz_dz[i,j,k] + self.a_z[k] * dsyz
                dsxy = dsxy/self.k_x[i] + self.memory_sxy_dx[i,j,k]
                dsyy = dsyy/self.k_y_half[j] + self.memory_syy_dy[i,j,k]
                dsyz = dsyz/self.k_z[k] + self.memory_syz_dz[i, j, k]
            self.vy[i,j,k]+=(dsxy+dsyy+dsyz)*dt/rho_half_y
        #update vz
        for i,j,k in ti.ndrange((star+1,nx-star),(star+1,ny-star),(star,nz-star-1)):
            x=(i-star)*dx
            y=(j-star)*dy
            z=(k-star)*dz+dz/2  
            rho_half_x_half_z=0.25*(self.rho[i,j,k]+self.rho[i,j+1,k]+self.rho[i+1,j+1,k]+self.rho[i+1,j,k])
            dsxzx=Dx3bm(self.sxz,i,j,k,self.c,star)/dx
            dsyzy=Dy3bm(self.syz,i,j,k,self.c,star)/dy
            dszz =Dz3fm(self.szz,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_sxz_dx[i,j,k] = self.b_x[i] * self.memory_sxz_dx[i,j,k] + self.a_x[i] * dsxzx
                self.memory_syz_dy[i,j,k] = self.b_y[j] * self.memory_syz_dy[i,j,k] + self.a_y[j] * dsyzy
                self.memory_szz_dz[i,j,k] =self. b_z_half[k] * self.memory_szz_dz[i,j,k] + self.a_z_half[k] * dszz
                dsxzx = dsxzx /self.k_x[i] + self.memory_sxz_dx[i,j,k]
                dsyzy = dsyzy /self.k_y[j] + self.memory_syz_dy[i,j,k]
                dszz  = dszz  /self.k_z_half[k] + self.memory_szz_dz[i,j,k]
            self.vz[i,j,k]+=(dszz+dsxzx+dsyzy)*dt/rho_half_x_half_z
    # implement Dirichlet boundary conditions on the six edges of the grid
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
            self.vz[i,j,k]=0  
    # source term  
        self.vz[isx+5,isy+5,isz+5]+=dt*Ricker(nt,dt,self.f0,1000)/self.rho[isx+5,isy+5,isz+5] 
       # update sxx szz,syy
        for i,j,k in ti.ndrange((star+1,nx-star),(star+1,ny-star),(star+1,nz-star)):
            x=(i-star)*dx
            y=(j-star)*dy
            z=(k-star)*dz
            lam_half_x=0.5*(self.lam[i,j,k]+self.lam[i+1,j,k])   
            mu_half_x =0.5*(self.mu[i,j,k]+self.mu[i+1,j,k])
            lam_plus_mu_half_x=lam_half_x+2*mu_half_x
            dvxx=Dx3bm(self.vx,i,j,k,self.c,star)/dx
            dvyy=Dy3bm(self.vy,i,j,k,self.c,star)/dy
            dvzz=Dz3bm(self.vz,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_dvx_dx[i,j,k] = self.b_x[i] * self.memory_dvx_dx[i,j,k] + self.a_x[i]*dvxx
                self.memory_dvy_dy[i,j,k] = self.b_y[j] * self.memory_dvy_dy[i,j,k] + self.a_y[j]*dvyy
                self.memory_dvz_dz[i,j,k] = self.b_z[k] * self.memory_dvz_dz[i,j,k] + self.a_z[k]*dvzz
                dvxx = dvxx /self.k_x[i] + self.memory_dvx_dx[i,j,k]
                dvyy = dvyy /self.k_y[j] + self.memory_dvy_dy[i,j,k]
                dvzz = dvzz /self.k_z[k] + self.memory_dvz_dz[i,j,k]
            self.sxx[i,j,k]+=(lam_plus_mu_half_x*dvxx+lam_half_x*(dvzz+dvyy))*dt
            self.syy[i,j,k]+=(lam_plus_mu_half_x*dvyy+lam_half_x*(dvxx+dvzz))*dt
            self.szz[i,j,k]+=(lam_plus_mu_half_x*dvzz+lam_half_x*(dvxx+dvyy))*dt   
        for i,j,k in ti.ndrange((star+1,nx-star),(star+1,ny-star),(star,nz-star)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy+dy/2
            z=(k-star)*dz
            mu_half_x_half_y=0.25*(self.mu[i,j+1,k]+self.mu[i,j,k]+self.mu[i,j,k]+self.mu[i+1,j,k])
            dvyx = Dx3fm(self.vy,i,j,k,self.c,star)/dx
            dvxy = Dy3fm(self.vx,i,j,k,self.c,star)/dy   
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml ):
                self.memory_dvy_dx[i,j,k] = self.b_x_half[i] * self.memory_dvy_dx[i,j,k]+ self.a_x_half[i] * dvyx 
                self.memory_dvx_dy[i,j,k] = self.b_y_half[j] * self.memory_dvx_dy[i,j,k]+ self.a_y_half[j] * dvxy 
                dvyx =dvyx /self.k_x_half[i] + self.memory_dvy_dx[i,j,k]
                dvxy =dvxy /self.k_y_half[j] + self.memory_dvx_dy[i,j,k] 
            self.sxy[i,j,k]+=mu_half_x_half_y*(dvxy+dvyx)*dt 

        for i,j,k in ti.ndrange((star,nx-star-1),(star,ny-star),(star,nz-star-1)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy
            z=(k-star)*dz+dz/2
            mu_half_x_half_z=0.25*(self.mu[i,j,k]+self.mu[i+1,j,k]+self.mu[i,j,k]+self.mu[i,j+1,k])
            dvzx = Dx3fm(self.vz,i,j,k,self.c,star)/dx
            dvxz = Dz3fm(self.vx,i,j,k,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml ):
                self.memory_dvz_dx[i,j,k] = self.b_x_half[i] * self.memory_dvz_dx[i,j,k]+ self.a_x_half[i] * dvzx 
                self.memory_dvx_dz[i,j,k] = self.b_z_half[k] * self.memory_dvx_dz[i,j,k]+ self.a_z_half[k] * dvxz 
                dvzx =dvzx /self.k_x_half[i] + self.memory_dvz_dx[i,j,k]
                dvxz =dvxz /self.k_z_half[k] + self.memory_dvx_dz[i,j,k] 
            self.sxz[i,j,k]+= mu_half_x_half_z*(dvxz+dvzx)*dt

        for i,j,k in ti.ndrange((star,nx-star),(star,ny-star-1),(star,nz-star-1)):
            x=(i-star)*dx
            y=(j-star)*dy+dy/2
            z=(k-star)*dz+dz/2
            mu_half_y_half_z=0.25*(self.mu[i,j+1,k]+self.mu[i,j,k]+self.mu[i,j,k+1]+self.mu[i,j,k])
            dvyz = Dz3fm(self.vy,i,j,k,self.c,star)/dz
            dvzy = Dy3fm(self.vz,i,j,k,self.c,star)/dy
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml ):
                self.memory_dvz_dy[i,j,k] = self.b_y_half[j] * self.memory_dvz_dy[i,j,k]+ self.a_y_half[j] *dvzy 
                self.memory_dvy_dz[i,j,k] = self.b_z_half[k] * self.memory_dvy_dz[i,j,k]+ self.a_z_half[k] *dvyz 
                dvzy =dvzy /self.k_y_half[j] + self.memory_dvz_dy[i,j,k]
                dvyz =dvyz /self.k_z_half[k] + self.memory_dvy_dz[i,j,k] 
            self.syz[i,j,k]+=mu_half_y_half_z*(dvyz+dvzy)*dt
        
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


    




    

