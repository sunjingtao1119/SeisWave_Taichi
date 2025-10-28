import taichi as ti
import numpy as np
from src.Differential3D_RSG3 import Drbm,Drfm
pi=np.pi

@ti.data_oriented
class ElasticWAVE:
    def __init__(self,
                 vs:ti.field, 
                 vp:ti.field,
                 rho:ti.field,
                 dx:ti.f32,
                 dy:ti.f32,
                 dz:ti.f32,
                 dt:ti.f32,
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
        self.xmin=0*dx
        self.ymin=0*dy
        self.zmin=0*dz
        self.nz=self.gridsize[0]
        self.nx=self.gridsize[1]
        self.ny=self.gridsize[2]
        self.star=accuracy
        self.c=self.diff_coff(accuracy)
        self.xmax=dx*(self.gridsize[0]-1-2*accuracy)-self.xmin    # Calculate the maximum x-coordinate based on grid size and spacing
        self.ymax=dy*(self.gridsize[1]-1-2*accuracy)-self.ymin    # Calculate the maximum z-coordinate based on grid size and spacing
        self.zmax=dz*(self.gridsize[2]-1-2*accuracy)-self.zmin    # Calculate the maximum z-coordinate based on grid size and spacing
        self.xmin_EAL=10*dx
        self.ymin_EAL=10*dy
        self.zmin_EAL=10*dz
        self.star=accuracy
        self.xmax_EAL=dx*(self.gridsize[0]-1-2*accuracy)-self.xmin_EAL    # Calculate the maximum x-coordinate based on grid size and spacing
        self.ymax_EAL=dy*(self.gridsize[1]-1-2*accuracy)-self.ymin_EAL    # Calculate the maximum z-coordinate based on grid size and spacing
        self.zmax_EAL=dz*(self.gridsize[2]-1-2*accuracy)-self.zmin_EAL    # Calculate the maximum z-coordinate based on grid size and spacing

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
        self.pml_x       =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[0]) 
        self.alpha_x     =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[0]) 
        self.k_x         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[0]) 
        self.b_x         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[0]) 
        self.a_x         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[0]) 
  

        self.pml_y       =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[1])    
        self.alpha_y     =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[1])    
        self.k_y         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[1])      
        self.b_y         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[1])   
        self.a_y         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[1])

        self.pml_z       =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[2])
        self.alpha_z     =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[2])
        self.k_z         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[2])
        self.b_z         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[2])
        self.a_z         =ti.Vector.field(n=2,dtype=fieldtype,shape=self.gridsize[2])

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
    def update_RSG(self,nt:int,source:ti.f32):
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
        # update vx vy vz
        for i,j,k in ti.ndrange((star,nx-star-1),(star+1,ny-star),(star+1,nz-star)):
            x=(i-star)*dx
            y=(j-star)*dy
            z=(k-star)*dz
            dsxxd1,dsxxd2,dsxxd3,dsxxd4=Drbm(self.sxx,i,j,k,self.c,star)
            dsxyd1,dsxyd2,dsxyd3,dsxyd4=Drbm(self.sxy,i,j,k,self.c,star)
            dsyyd1,dsyyd2,dsyyd3,dsyyd4=Drbm(self.syy,i,j,k,self.c,star)
            dsyzd1,dsyzd2,dsyzd3,dsyzd4=Drbm(self.syz,i,j,k,self.c,star)
            dsxzd1,dsxzd2,dsxzd3,dsxzd4=Drbm(self.sxz,i,j,k,self.c,star)
            dszzd1,dszzd2,dszzd3,dszzd4=Drbm(self.szz,i,j,k,self.c,star)        

            dsxxdx =(dsxxd1+dsxxd2-dsxxd3-dsxxd4)/(4*dx)
            dsxydy =(dsxyd1-dsxyd2+dsxyd3-dsxyd4)/(4*dy)
            dsxzdz =(dsxzd1+dsxzd2+dsxzd3+dsxzd4)/(4*dz)
            
            dsxydx =(dsxyd1+dsxyd2-dsxyd3-dsxyd4)/(4*dx)
            dsyydy =(dsyyd1-dsyyd2+dsyyd3-dsyyd4)/(4*dy)
            dsyzdz =(dsyzd1+dsyzd2+dsyzd3+dsyzd4)/(4*dz)
            
            dsxzdx =(dsxzd1+dsxzd2-dsxzd3-dsxzd4)/(4*dx)
            dsyzdy =(dsyzd1-dsyzd2+dsyzd3-dsyzd4)/(4*dy)
            dszzdz =(dszzd1+dszzd2+dszzd3+dszzd4)/(4*dz)

            rho=(self.rho[i,j,k]+self.rho[i+1,j,k]+self.rho[i+1,j+1,k]+self.rho[i,j+1,k]
                  +self.rho[i,j,k+1]+self.rho[i+1,j,k+1]+self.rho[i+1,j+1,k+1]+self.rho[i,j+1,k+1])/8          

            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_sxx_dx[i,j,k] = self.b_x[i][0] * self.memory_sxx_dx[i,j,k] + self.a_x[i][0] * dsxxdx
                self.memory_sxy_dy[i,j,k] = self.b_y[j][0] * self.memory_sxy_dy[i,j,k] + self.a_y[j][0] * dsxydy
                self.memory_sxz_dz[i,j,k] = self.b_z[k][0] * self.memory_sxz_dz[i,j,k] + self.a_z[k][0] * dsxzdz

                self.memory_sxy_dx[i,j,k] = self.b_x[i][0] * self.memory_sxy_dx[i,j,k] + self.a_x[i][0] * dsxydx
                self.memory_syy_dy[i,j,k] = self.b_y[j][0] * self.memory_syy_dy[i,j,k] + self.a_y[j][0] * dsyydy
                self.memory_syz_dz[i,j,k] = self.b_z[k][0] * self.memory_syz_dz[i,j,k] + self.a_z[k][0] * dsyzdz

                self.memory_sxz_dx[i,j,k] = self.b_x[i][0] * self.memory_sxz_dx[i,j,k] + self.a_x[i][0] * dsxzdx
                self.memory_syz_dy[i,j,k] = self.b_y[j][0] * self.memory_syz_dy[i,j,k] + self.a_y[j][0] * dsyzdy
                self.memory_szz_dz[i,j,k] = self.b_z[k][0] * self.memory_szz_dz[i,j,k] + self.a_z[k][0] * dszzdz
                dsxxdx = dsxxdx/self.k_x[i][0] + self.memory_sxx_dx[i,j,k]
                dsxydy = dsxydy/self.k_y[j][0] + self.memory_sxy_dy[i,j,k]
                dsxzdz = dsxzdz/self.k_z[k][0] + self.memory_sxz_dz[i,j,k]

                dsxydx = dsxydx/self.k_x[i][0] + self.memory_sxy_dx[i,j,k]
                dsyydy = dsyydy/self.k_y[j][0] + self.memory_syy_dy[i,j,k]
                dsyzdz = dsyzdz/self.k_z[k][0] + self.memory_syz_dz[i,j,k]

                dsxzdx = dsxzdx/self.k_x[i][0] + self.memory_sxz_dx[i,j,k]
                dsyzdy = dsyzdy/self.k_y[j][0] + self.memory_syz_dy[i,j,k]
                dszzdz = dszzdz/self.k_z[k][0] + self.memory_szz_dz[i,j,k] 
                self.vx[i,j,k]+=(dsxxdx+dsxydy+dsxzdz)*dt/ rho
                self.vy[i,j,k]+=(dsxydx+dsyydy+dsyzdz)*dt/ rho 
                self.vz[i,j,k]+=(dsxzdx+dsyzdy+dszzdz)*dt/ rho 

            else:
                self.vx[i,j,k]+=(dsxxdx+dsxydy+dsxzdz)*dt/ rho
                self.vy[i,j,k]+=(dsxydx+dsyydy+dsyzdz)*dt/ rho 
                self.vz[i,j,k]+=(dsxzdx+dsyzdy+dszzdz)*dt/ rho      

    # source term  
        for i,j,k in ti.ndrange((0,2),(0,2),(0,2)):
            self.sxx[isx+i,isy+j,isz+k]-=source/8
            self.szz[isx+i,isy+j,isz+k]-=source/8
            self.syy[isx+i,isy+j,isz+k]-=source/8

       # update sxx szz,syy
        for i,j,k in ti.ndrange((star+1,nx-star),(star+1,ny-star),(star+1,nz-star)):
            x=(i-star)*dx+dx/2
            y=(j-star)*dy+dy/2
            z=(k-star)*dz+dz/2
            lam=self.lam[i,j,k]  
            mu =self.mu[i,j,k]
            lam_plus_2mu=lam+2*mu
            dvxd1, dvxd2, dvxd3, dvxd4=Drfm(self.vx,i,j,k,self.c,star)
            dvyd1, dvyd2, dvyd3, dvyd4=Drfm(self.vy,i,j,k,self.c,star)
            dvzd1, dvzd2, dvzd3, dvzd4=Drfm(self.vz,i,j,k,self.c,star)


            dvxdx=(dvxd1+dvxd2-dvxd3-dvxd4) /(4*dx)
            dvxdy=(dvxd1-dvxd2+dvxd3-dvxd4) /(4*dy)
            dvxdz=(dvxd1+dvxd2+dvxd3+dvxd4) /(4*dz)

            dvydx=(dvyd1+dvyd2-dvyd3-dvyd4) /(4*dx)
            dvydy=(dvyd1-dvyd2+dvyd3-dvyd4) /(4*dy)
            dvydz=(dvyd1+dvyd2+dvyd3+dvyd4) /(4*dz) 

            dvzdx=(dvzd1+dvzd2-dvzd3-dvzd4) /(4*dx)
            dvzdy=(dvzd1-dvzd2+dvzd3-dvzd4) /(4*dy)
            dvzdz=(dvzd1+dvzd2+dvzd3+dvzd4) /(4*dz)
        
            if (x<=xmin_pml or x>=xmax_pml or y<=ymin_pml or y>=ymax_pml or z<=zmin_pml or z>=zmax_pml):
    
                self.memory_dvx_dx[i,j,k] = self.b_x[i][1] * self.memory_dvx_dx[i,j,k] + self.a_x[i][1]*dvxdx
                self.memory_dvy_dy[i,j,k] = self.b_y[j][1] * self.memory_dvy_dy[i,j,k] + self.a_y[j][1]*dvydy
                self.memory_dvz_dz[i,j,k] = self.b_z[k][1] * self.memory_dvz_dz[i,j,k] + self.a_z[k][1]*dvzdz

                self.memory_dvy_dx[i,j,k] = self.b_x[i][1] * self.memory_dvy_dx[i,j,k] + self.a_x[i][1]*dvydx 
                self.memory_dvx_dy[i,j,k] = self.b_y[j][1] * self.memory_dvx_dy[i,j,k] + self.a_y[j][1]*dvxdy 

                self.memory_dvz_dx[i,j,k] = self.b_x[i][1] * self.memory_dvz_dx[i,j,k] + self.a_x[i][1]*dvzdx 
                self.memory_dvx_dz[i,j,k] = self.b_z[k][1] * self.memory_dvx_dz[i,j,k] + self.a_z[k][1]*dvxdz

                self.memory_dvz_dy[i,j,k] = self.b_y[j][1] * self.memory_dvz_dy[i,j,k] + self.a_y[j][1]*dvzdy 
                self.memory_dvy_dz[i,j,k] = self.b_z[k][1] * self.memory_dvy_dz[i,j,k] + self.a_z[k][1]*dvydz 

                dvxdx = dvxdx /self.k_x[i][1] + self.memory_dvx_dx[i,j,k]
                dvydy = dvydy /self.k_y[j][1]+ self.memory_dvy_dy[i,j,k]
                dvzdz = dvzdz /self.k_z[k][1] + self.memory_dvz_dz[i,j,k]

                dvydx = dvydx /self.k_x[i][1] + self.memory_dvy_dx[i,j,k]
                dvxdy = dvxdy /self.k_y[j][1] + self.memory_dvx_dy[i,j,k]

                dvzdx = dvzdx /self.k_x[i][1] + self.memory_dvz_dx[i,j,k]
                dvxdz = dvxdz /self.k_z[k][1] + self.memory_dvx_dz[i,j,k]

                dvzdy = dvzdy /self.k_y[j][1] + self.memory_dvz_dy[i,j,k]
                dvydz = dvydz /self.k_z[k][1] + self.memory_dvy_dz[i,j,k]
                self.sxx[i,j,k]+=(lam_plus_2mu*dvxdx+lam*(dvzdz+dvydy))*dt
                self.syy[i,j,k]+=(lam_plus_2mu*dvydy+lam*(dvxdx+dvzdz))*dt
                self.szz[i,j,k]+=(lam_plus_2mu*dvzdz+lam*(dvxdx+dvydy))*dt 
                self.sxy[i,j,k]+= mu*(dvxdy+dvydx)*dt 
                self.sxz[i,j,k]+= mu*(dvxdz+dvzdx)*dt 
                self.syz[i,j,k]+= mu*(dvydz+dvzdy)*dt 
            else:
                self.sxx[i,j,k]+=(lam_plus_2mu*dvxdx+lam*(dvzdz+dvydy))*dt
                self.syy[i,j,k]+=(lam_plus_2mu*dvydy+lam*(dvxdx+dvzdz))*dt
                self.szz[i,j,k]+=(lam_plus_2mu*dvzdz+lam*(dvxdx+dvydy))*dt 
                self.sxy[i,j,k]+= mu*(dvxdy+dvydx)*dt 
                self.sxz[i,j,k]+= mu*(dvxdz+dvzdx)*dt 
                self.syz[i,j,k]+= mu*(dvydz+dvzdy)*dt 
      
     
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
    @staticmethod
    def diff_op(order:int):
        if order==5:
            c=ti.field(dtype=ti.f32,shape=(order,))
            c[0]=1.2429701
            c[1]=-1.134665e-1
            c[2]=2.685699e-2
            c[3]=-6.762350e-3
            c[4]=1.164592e-3

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
            self.pml_x[i][0]= pml_dx_temp
            self.alpha_x[i][0]=alpha_x_temp
            self.k_x[i][0]=k_x_temp
            if self.alpha_x[i][0]<0:
                self.alpha_x[i][0]=0
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
            self.pml_x[i][1]= pml_half_x_temp
            self.alpha_x[i][1]=alpha_half_x_temp
            self.k_x[i][1]=k_half_x_temp
            if self.alpha_x[i][1]<0:
                self.alpha_x[i][1]=0
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
            self.pml_y[i][0]= pml_y_temp
            self.alpha_y[i][0]=alpha_y_temp
            self.k_y[i][0]=k_y_temp
            if self.alpha_y[i][0]<0:
                self.alpha_y[i][0]=0
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
            self.pml_y[i][1]= pml_half_y_temp
            self.alpha_y[i][1]=alpha_half_y_temp
            self.k_y[i][1]=k_half_y_temp
            if self.alpha_y[i][1]<0:
                self.alpha_y[i][1]=0
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
            self.pml_z[i][0]= pml_z_temp
            self.alpha_z[i][0]=alpha_z_temp
            self.k_z[i][0]=k_z_temp
            if self.alpha_z[i][0]<0:
                self.alpha_z[i][0]=0
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
            self.pml_z[i][1]= pml_half_z_temp
            self.alpha_z[i][1]=alpha_half_z_temp
            self.k_z[i][1]=k_half_z_temp
            if self.alpha_z[i][1]<0:
                self.alpha_z[i][1]=0

        # PML damping parameters for time steps of memory variable 
        for i in range(nx):
            self.b_x[i][0] =(1-(1-theta)*dt*(self.pml_x[i][0]/self.k_x[i][0] + self.alpha_x[i][0]))/(1+theta*dt*(self.pml_x[i][0]/self.k_x[i][0] + self.alpha_x[i][0]))
            self.b_x[i][1] =(1-(1-theta)*dt*(self.pml_x[i][1]/self.k_x[i][1] + self.alpha_x[i][1]))/(1+theta*dt*(self.pml_x[i][1]/self.k_x[i][1] + self.alpha_x[i][1]))
            if np.abs(self.pml_x[i][0])>1e-6:
                self.a_x[i][0] = - dt*self.pml_x[i][0]/(self.k_x[i][0]* self.k_x[i][0])/(1+theta*dt*(self.pml_x[i][0]/self.k_x[i][0] + self.alpha_x[i][0]))
            if np.abs(self.pml_x[i][1])>1e-6:
                self.a_x[i][1] = - dt*self.pml_x[i][1]/(self.k_x[i][1]* self.k_x[i][1])/(1+theta*dt*(self.pml_x[i][1]/self.k_x[i][1] + self.alpha_x[i][1]))           
        for i in range(ny):
            self.b_y[i][0] = (1-(1-theta)*dt*(self.pml_y[i][0]/self.k_y[i][0] + self.alpha_y[i][0]))/(1+theta*dt*(self.pml_y[i][0]/self.k_y[i][0] + self.alpha_y[i][0]))
            self.b_y[i][1] = (1-(1-theta)*dt*(self.pml_y[i][1]/self.k_y[i][1] + self.alpha_y[i][1]))/(1+theta*dt*(self.pml_y[i][1]/self.k_y[i][1] + self.alpha_y[i][1]))
            if np.abs(self.pml_y[i][0]) > 1e-6:
                self.a_y[i][0] = -dt * self.pml_y[i][0]/ (self.k_y[i][0] * self.k_y[i][0]) / (1 + theta * dt * (self.pml_y[i][0]/ self.k_y[i][0] + self.alpha_y[i][0]))
            if np.abs(self.pml_y[i][1]) > 1e-6:
                self.a_y[i][1] = -dt * self.pml_y[i][1]/ (self.k_y[i][1] * self.k_y[i][1]) / (1 + theta * dt * (self.pml_y[i][1] / self.k_y[i][1] + self.alpha_y[i][1]))
        
        for i in range(nz):
            self.b_z[i][0] = (1-(1-theta)*dt*( self.pml_z[i][0]/self.k_z[i][0] + self.alpha_z[i][0]))/(1+theta*dt*(self.pml_z[i][0]/self.k_z[i][0] + self.alpha_z[i][0]))
            self.b_z[i][1] = (1-(1-theta)*dt*(self.pml_z [i][1]/self.k_z[i][1] + self.alpha_z[i][1]))/(1+theta*dt*(self.pml_z[i][1]/self.k_z[i][1] + self.alpha_z[i][1])) 
            if np.abs(self.pml_z[i][0])>1e-6:
                self.a_z[i][0] = - dt*self.pml_z[i][0]/(self.k_z[i][0]* self.k_z[i][0])/(1+theta*dt*(self.pml_z[i][0]/self.k_z[i][0] + self.alpha_z[i][0]))
            if np.abs(self.pml_z[i][1])>1e-6:
                self.a_z[i][1] = - dt*self.pml_z[i][1]/(self.k_z[i][1]* self.k_z[i][1])/(1+theta*dt*(self.pml_z[i][1]/self.k_z[i][1] + self.alpha_z[i][1]))   


    




    

