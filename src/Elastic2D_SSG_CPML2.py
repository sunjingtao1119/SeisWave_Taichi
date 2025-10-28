import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from src.Differential2D import Dx2bm,Dx2fm,Dz2bm,Dz2fm
from src.BaseFun import Ricker, Ricker2
pi=np.pi

@ti.data_oriented
class ElasticWAVE:
    def __init__(self,
                 vs:ti.field, 
                 vp:ti.field,
                 rho:ti.field,
                 dx:float,
                 dz:float,
                 dt:float,
                 isx:int,
                 isz:int,
                 rsx:int,
                 rsz:int,
                 nt:int,
                 accuracy=3,
                 freq=100.,
                 fieldtype=ti.f32):
        # Initialize model parameters
        self.vs=vs
        self.vp=vp
        self.rho=rho
        self.gridsize=vs.shape
        self.dx=dx
        self.dz=dz
        self.xmin=0
        self.zmin=0
        self.star=accuracy
        self.c=self.diff_coff(accuracy)
        self.xmax=self.xmin+dx*(self.gridsize[0]-1-2*accuracy)    # Calculate the maximum x-coordinate based on grid size and spacing
        self.zmax=self.zmin+dz*(self.gridsize[1]-1-2*accuracy)    # Calculate the maximum z-coordinate based on grid size and spacing
        self.mu =self.Compute_mu(fieldtype)
        self.lam=self.Compute_lam(fieldtype)
        # source term
        self.f0=freq
        self.dt=dt
        self.isx=isx
        self.isz=isz
        self.rsx=rsx
        self.rsz=rsz
        self.nt=nt
        self.src =ti.field(dtype=fieldtype,shape=nt)
        self.data=ti.field(dtype=fieldtype,shape=nt)
        # velocity field and stress field initial 
        self.vx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.vz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.sxz=ti.field(dtype=fieldtype,shape=self.gridsize)
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
    
        self.pml_z       =ti.field(fieldtype,shape=self.gridsize[1])
        self.pml_z_half  =ti.field(fieldtype,shape=self.gridsize[1])
        self.alpha_z     =ti.field(fieldtype,shape=self.gridsize[1])
        self.alpha_z_half=ti.field(fieldtype,shape=self.gridsize[1])
        self.k_z         =ti.field(fieldtype,shape=self.gridsize[1])
        self.k_z_half    =ti.field(fieldtype,shape=self.gridsize[1])
        self.b_z         =ti.field(fieldtype,shape=self.gridsize[1])
        self.b_z_half    =ti.field(fieldtype,shape=self.gridsize[1])
        self.a_z         =ti.field(fieldtype,shape=self.gridsize[1])
        self.a_z_half    =ti.field(fieldtype,shape=self.gridsize[1])
        self.xmin_pml=self.xmin
        self.xmax_pml=self.xmax
        self.zmin_pml=self.zmin
        self.zmax_pml=self.zmax
        # Initialize memory variables for PML (Perfectly Matched Layer)
        self.memory_sxx_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_sxx_dz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_sxz_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_sxz_dz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_szz_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_szz_dz=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvx_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
        self.memory_dvx_dz=ti.field(dtype=fieldtype,shape=self.gridsize) 
        self.memory_dvz_dx=ti.field(dtype=fieldtype,shape=self.gridsize)
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
    def init_src(self,src_type:int,src_scale:ti.f32):
        dt=self.dt
        f0=self.f0
        if src_type==1:
            for i in ti.ndrange(self.nt):
                self.src[i]=Ricker(i,dt,f0,src_scale)
        if src_type==2:
            for i in ti.ndrange(self.nt):
                self.src[i]=Ricker2(i,dt,f0,src_scale)
    
    # Freedome boundary condtion 
    @ti.kernel
    def AEA_init(self):
        nz=self.gridsize[1]
        order=self.star
        for i,j in ti.ndrange(order,nz):
            self.lam[i,j]=0
            self.rho[i,j]=0.5*self.rho[i,j]
    # Freedome boundary condtion 
    @ti.kernel
    def AEA(self):
        order=self.star
        for j in ti.ndrange(self.gridsize[1]):
            self.sxx[order,j]=0

    # Update wavefield  staggered grid method method one
    @ti.kernel
    def update_SSG_VS(self,nt:int):   
        star=self.star
        dx=self.dx
        dz=self.dz
        dt=self.dt
        isz=self.isz
        isx=self.isx
        xmin_pml = self.xmin_pml
        xmax_pml = self.xmax_pml
        zmin_pml = self.zmin_pml
        zmax_pml = self.zmax_pml
        nx=self.gridsize[0]
        nz=self.gridsize[1]
        # update vx
        for i,j in ti.ndrange((star+1,nx-star),(star+1,nz-star)):
            x=(i-star)*dx
            z=(j-star)*dz
            dsxxdx=Dz2bm(self.sxx,i,j,self.c,star)/dx
            dsxzdz=Dx2bm(self.sxz,i,j,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_sxx_dx[i,j] = self.b_x[i] * self.memory_sxx_dx[i,j] + self.a_x[i] * dsxxdx
                self.memory_sxz_dz[i,j] = self.b_z[j] * self.memory_sxz_dz[i,j] + self.a_z[j] * dsxzdz
                dsxxdx = dsxxdx/self.k_x[i] + self.memory_sxx_dx[i,j]
                dsxzdz = dsxzdz/self.k_z[j] + self.memory_sxz_dz[i,j]
            self.vx[i,j]+=(dsxxdx+dsxzdz)*dt/self.rho[i,j]      
        # update vz
        for i,j in ti.ndrange((star,nx-star-1),(star,nz-star-1)):
            x=(i-star)*dx+dx/2
            z=(j-star)*dz+dz/2
            rho_half_x_half_z=0.25*(self.rho[i,j]+self.rho[i,j+1]+self.rho[i+1,j+1]+self.rho[i+1,j])
            dsxzdx=Dz2fm(self.sxz,i,j,self.c,star)/dx
            dszzdz=Dx2fm(self.szz,i,j,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_sxz_dx[i,j] =self.b_x_half[i] * self.memory_sxz_dx[i,j] + self.a_x_half[i] * dsxzdx
                self.memory_szz_dz[i,j] =self.b_z_half[j] * self.memory_szz_dz[i,j] + self.a_z_half[j] * dszzdz
                dsxzdx = dsxzdx /self.k_x_half[i] + self.memory_sxz_dx[i,j]
                dszzdz = dszzdz /self.k_z_half[j] + self.memory_szz_dz[i,j]
            self.vz[i,j]+=(dszzdz+dsxzdx)*dt/rho_half_x_half_z
       # add source

        #self.sxx[isx,isz]+=self.src[nt]
        self.szz[isx,isz]+=self.src[nt] 
       # update sxx szz
        for i,j in ti.ndrange((star,nx-star-1),(star+1,nz-star)):
            x=(i-star)*dx+dx/2
            z=(j-star)*dz
            lam_half_x=0.5*(self.lam[i,j]+self.lam[i+1,j])
            mu_half_x =0.5*(self.mu[i,j]+self.mu[i+1,j])
            lam_plus_mu_half_x=lam_half_x+2*mu_half_x
            dvxdx=Dz2fm(self.vx,i,j,self.c,star)/dx
            dvzdz=Dx2bm(self.vz,i,j,self.c,star)/dz
            if (x<=xmin_pml or x>=xmax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_dvx_dx[i,j] = self.b_x_half[i] * self.memory_dvx_dx[i,j] + self.a_x_half[i]*dvxdx
                self.memory_dvz_dz[i,j] = self.b_z[j]      * self.memory_dvz_dz[i,j] + self.a_z[j]     *dvzdz
                dvxdx = dvxdx /self.k_x_half[i] + self.memory_dvx_dx[i,j]
                dvzdz = dvzdz /self.k_z[j]      + self.memory_dvz_dz[i,j]
            self.sxx[i,j]+=(lam_plus_mu_half_x*dvxdx+lam_half_x*dvzdz)*dt
            self.szz[i,j]+=(lam_plus_mu_half_x*dvzdz+lam_half_x*dvxdx)*dt    
    # update sxz
        for i,j in ti.ndrange((star+1,nx-star),(star,nz-star-1)):
            x=(i-star)*dx
            z=(j-star)*dz+dz/2
            mu_half_z=0.5*(self.mu[i,j+1]+self.mu[i,j])
            dvzdx=Dz2bm(self.vz,i,j,self.c,star)/dx
            dvxdz=Dx2fm(self.vx,i,j,self.c,star)/dz  
            if (x<=xmin_pml or x>=xmax_pml or z<=zmin_pml or z>=zmax_pml):
                self.memory_dvz_dx[i,j] =self.b_x[i] * self.memory_dvz_dx[i,j]+ self.a_x[i] * dvzdx 
                self.memory_dvx_dz[i,j] =self.b_z_half[j] * self.memory_dvx_dz[i,j]+ self.a_z_half[j]*dvxdz 
                dvzdx =dvzdx /self.k_x[i] + self.memory_dvz_dx[i,j]
                dvxdz =dvxdz /self.k_z_half[j] + self.memory_dvx_dz[i,j] 
            self.sxz[i,j]+=mu_half_z*(dvxdz+dvzdx)*self.dt
        self.data[nt]=self.vx[self.rsx,self.rsz]   # Receiver trace data
    

    
    @staticmethod
    def diff_coff(order:int):
        b=np.zeros((order))
        b[0]=1
        A=np.zeros((order,order))
        for i in range(order):
            for j in range(order):
                A[i,j]=(2*j+1)**(2*i+1)
        c_np=np.linalg.solve(A,b)  # Calculate the finite difference coefficients
        c=ti.field(dtype=ti.f32,shape=order)
        c.from_numpy(c_np)
        return c
    
    @staticmethod
    def diff_coff_op(order:int):
        size=int(order)
        c=ti.field(dtype=ti.f32,shape=size)
        if order==2:
            c[0]=1.129042
            c[1]=-0.0430142
        elif order==3:
            c[0]=2.081695391597685
            c[1]=-0.01139368190892791
            c[2]=-0.2095028691741802
        elif order==4:
            c[0]=1.231666
            c[1]=-1.041182e-1
            c[2]=2.063707e-2  
            c[3]=-3.570998e-3  
        elif order==5:
            c[0]=1.236425
            c[1]=-0.10811
            c[2]=0.02339911  
            c[3]=-0.005061550                 
            c[4]=0.0007054313 
        return c
        
    def SetADEPML2D(self,pml_surface,parameter:dict):
        vp_max=parameter["vp_max"]  
        dt=self.dt
        dx=self.dx
        dz=self.dz
        nx=self.gridsize[0]
        nz=self.gridsize[1]
        xmin=self.xmin
        xmax=self.xmax
        zmin=self.zmin
        zmax=self.zmax
        pml_x_thick=parameter["pml_x_thick"]    
        pml_z_thick=parameter["pml_z_thick"]
        Rcoef =parameter["Rcoef"]
        alpha_max_pml=parameter["alpha_max_pml"]
        k_max_pml=parameter["kmax_pml"]
        theta=parameter["theta"]
        star=self.star
        d0_x = -3.0 * vp_max * ti.log(Rcoef) / (2.0 * pml_x_thick)
        d0_z = -3.0 * vp_max * ti.log(Rcoef) / (2.0 * pml_z_thick)
        # set pml boudary
        Use_PML_x_Left  = pml_surface[0]   
        Use_PML_X_Right = pml_surface[1]
        Use_PML_Z_Up    = pml_surface[2]
        Use_PML_Z_Bottom= pml_surface[3]
        # The location of pml boundary
        self.xmin_pml = xmin + pml_x_thick
        self.xmax_pml = xmax - pml_x_thick
        self.zmin_pml = zmin + pml_z_thick
        self.zmax_pml = zmax - pml_z_thick
        # x direction 
        # define damping profile at the grid point  
        for i in range(nx):
            x=(i-star)*dx
            if (x<= self.xmin_pml and Use_PML_x_Left) :
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
            if (x_half<= self.xmin_pml and Use_PML_x_Left) :
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

        for i in range(nz):
            self.b_z[i]      = (1-(1-theta)*dt*(     self.pml_z[i]/self.k_z[i] + self.alpha_z[i]))/(1+theta*dt*(self.pml_z[i]/self.k_z[i] + self.alpha_z[i]))
            self.b_z_half[i] = (1-(1-theta)*dt*(self.pml_z_half[i]/self.k_z_half[i] +self.alpha_z_half[i]))/(1+theta*dt*(self.pml_z_half[i]/self.k_z_half[i]+ self.alpha_z_half[i])) 
            if np.abs(self.pml_z[i])>1e-6:
                self.a_z[i] = - dt*self.pml_z[i]/(self.k_z[i]* self.k_z[i])/(1+theta*dt*(self.pml_z[i]/self.k_z[i] + self.alpha_z[i]))
            if np.abs(self.pml_z_half[i])>1e-6:
                self.a_z_half[i] = - dt*self.pml_z_half[i]/(self.k_z_half[i]* self.k_z_half[i])/(1+theta*dt*(self.pml_z_half[i]/self.k_z_half[i] + self.alpha_z_half[i]))   
    
 

    
    




    

