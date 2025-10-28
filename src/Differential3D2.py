import taichi as ti
'''
  c is ti.field and used to store the n-point stencil differece coefficient
'''
## 3D differential Staggered Grid ###
# Forward difference
@ti.func
def Dx3fm(fieldx,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[k,i+id+1, j] - fieldx[k,i-id, j])
    return value
@ti.func
def Dy3fm(fieldy,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[k,i, j+id+1] - fieldy[k,i, j-id])
    return value
@ti.func
def Dz3fm(fieldz,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[k+id+1,i, j] - fieldz[k-id,i, j])
    return value
# Backward difference
@ti.func
def Dx3bm(fieldx,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[k,i+id, j] - fieldx[k,i-id-1, j])
    return value
@ti.func
def Dy3bm(fieldy,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[k,i, j+id] - fieldy[k,i, j-id-1])
    return value
    
@ti.func
def Dz3bm(fieldz,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[k+id,i, j] - fieldz[k-id-1,i, j])
    return value



