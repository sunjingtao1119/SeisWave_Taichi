import taichi as ti
## 2D differential Staggered Grid###
# Forward difference
@ti.func
def Dx1fm(fieldx,i,coff,n):
    value=0.
    for id in range(n):
        value += coff[id]*(fieldx[i+id+1] - fieldx[i-id])
    return value
# Backward difference
@ti.func
def Dx1bm(fieldx,i,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[i+id] - fieldx[i-id-1])
    return value
@ti.func
def Mx1bm(fieldx,i):
    value=0.
    value +=(-3)*(fieldx[i+1]-fieldx[i])+(fieldx[i+2]-fieldx[i-1])
    return value







