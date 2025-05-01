import taichi as ti
'''
  c is ti.field and used to store the n-point stencil differece coefficient
'''
## 3D differential Staggered Grid ###
# Forward difference
@ti.func
def Dx3fm(fieldx,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[i+id+1, j,k] - fieldx[i-id, j,k])
    return value
@ti.func
def Dy3fm(fieldy,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[i, j+id+1,k] - fieldy[i, j-id,k])
    return value
@ti.func
def Dz3fm(fieldz,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i, j,k+id+1] - fieldz[i, j,k-id])
    return value
# Backward difference
@ti.func
def Dx3bm(fieldx,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[i+id, j,k] - fieldx[i-id-1, j,k])
    return value
@ti.func
def Dy3bm(fieldy,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[i, j+id,k] - fieldy[i, j-id-1,k])
    return value
    
@ti.func
def Dz3bm(fieldz,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i, j,k+id] - fieldz[i, j,k-id-1])
    return value
## 3D differential Lebedev Staggered Grid ###
# Forward difference
@ti.func
def Drd1fm(fieldx,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[i+id+1, j+id+1,k+id+1] - fieldx[i-id, j-id,k-id])
    return value
@ti.func
def Drd2fm(fieldy,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[i+id+1, j+id+1,k-id] - fieldy[i-id, j-id,k+id+1])
    return value
@ti.func
def Drd3fm(fieldz,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i+id+1, j-id,k+id+1] - fieldz[i-id, j+id+1,k-id])
    return value
@ti.func
def Drd4fm(fieldz,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i+id+1, j-id,k-id] - fieldz[i-id, j+id+1,k+id+1])
    return value
# Backward difference
@ti.func
def Drd1bm(fieldx,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[i+id, j+id,k+id] - fieldx[i-id-1, j-id-1,k-id-1])
    return value
@ti.func
def Drd2bm(fieldy,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[i+id, j+id,k-id-1] - fieldy[i-id-1, j-id-1,k+id])
    return value
@ti.func
def Drd3bm(fieldz,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i+id, j-id-1,k+id] - fieldz[i-id-1, j+id,k-id-1])
    return value
@ti.func
def Drd4bm(fieldz,i,j,k,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i+id, j-id-1,k-id-1] - fieldz[i-id-1, j+id,k+id])
    return value


