import taichi as ti
import math
pi=math.pi
@ti.kernel
def Ricker(src:ti.template(),nt:int,dt:float,freq:float,src_scale:float):
    for i in range(nt):
        t0=1/freq
        a=pi**2*freq**2
        t=i*dt
        tau = t-t0
        src[i]= -src_scale*2*a*tau*ti.exp(-a*tau**2 )


@ti.kernel
def Ricker2(src:ti.template(),nt:int,dt:float,freq:float,src_scale:float):
    for i in range(nt):
        t0=1/freq
        a=pi**2*freq**2
        t=i*dt
        tau = t-t0
        src[i]= src_scale*(1 -2* a*tau*tau )*ti.exp(-a*tau*tau  )

  
