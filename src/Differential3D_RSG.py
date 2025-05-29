import taichi as ti
'''
  c is ti.field and used to store the n-point stencil differece coefficient
'''
## 3D differential Lebedev Staggered Grid ###
# Forward difference
@ti.func
def Drd1fm(fieldx,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[k+id+1,i+id+1, j+id+1] - fieldx[k-id,i-id, j-id])
    return value
@ti.func
def Drd2fm(fieldy,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[k-id,i+id+1, j+id+1] - fieldy[k+id+1,i-id, j-id])
    return value
@ti.func
def Drd3fm(fieldz,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[k+id+1,i+id+1, j-id] - fieldz[k-id,i-id, j+id+1])
    return value
@ti.func
def Drd4fm(fieldz,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[k-id,i+id+1, j-id] - fieldz[k+id+1,i-id, j+id+1])
    return value
# Backward difference
@ti.func
def Drd1bm(fieldx,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[k+id,i+id, j+id] - fieldx[k-id-1,i-id-1, j-id-1])
    return value
@ti.func
def Drd2bm(fieldy,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[k-id-1,i+id, j+id] - fieldy[k+id,i-id-1, j-id-1])
    return value
@ti.func
def Drd3bm(fieldz,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[k+id,i+id, j-id-1] - fieldz[k-id-1,i-id-1, j+id])
    return value
@ti.func
def Drd4bm(fieldz,k,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[k-id-1,i+id, j-id-1] - fieldz[k+id,i-id-1, j+id])
    return value
