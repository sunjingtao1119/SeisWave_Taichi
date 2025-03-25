import taichi as ti
import math
pi=math.pi
@ti.func
def Ricker(nt:int,dt:float,freq:float,src_scale:float)->ti.f32:
    t0=1.5/freq
    a=pi**2*freq**2
    t=nt*dt
    tau = t-t0
    src= -src_scale*2*a*tau*ti.exp(-a*tau**2 )
    return src

@ti.func
def Ricker2(nt:int,dt:float,t0:float,freq:float,src_scale:float)->ti.f32:
    a=pi**2*freq**2
    t=nt*dt
    tau = t-t0
    src= src_scale*(1 -2* a*tau**2 )*ti.exp(-a*tau**2  )
    return src

@ti.func
def Ricker3(nt:int,dt:float,t0:float,freq:float,src_scale:float)->ti.f32:
    ts = 1.0 / freq
    t=nt*dt
    tau = pi*(t-1.5*ts-t0)/ts
    src =  src_scale*(1 -2*tau**2 )*ti.exp(-tau**2  )
    return src


