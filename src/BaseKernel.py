import taichi as ti
import math
pi=math.pi
@ti.kernel
def Ricker(nt:int,dt:float,freq:float,src_scale:float)->ti.f32:
    t0=1./freq
    a=pi**2*freq**2
    t=nt*dt
    tau = t-t0
    src= -src_scale*2*a*tau*ti.exp(-a*tau**2 )
    return src

@ti.kernel
def Ricker2(nt:int,dt:float,t0:float,freq:float,src_scale:float)->ti.f32:
    a=pi**2*freq**2
    t=nt*dt
    tau = t-t0
    src= src = src_scale*(1 -2* a*tau*tau )*ti.exp(-a*tau**2  )
    return src
  
