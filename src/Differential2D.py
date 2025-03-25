import taichi as ti
## 2D differential Staggered Grid###
# Forward difference
@ti.func
def Dx2fm(fieldx,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id]*(fieldx[i+id+1,j] - fieldx[i-id, j])
    return value
@ti.func
def Dz2fm(fieldz,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i, j+id+1] - fieldz[i, j-id])
    return value
# Backward difference
@ti.func
def Dx2bm(fieldx,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldx[i+id, j] - fieldx[i-id-1, j])
    return value
@ti.func
def Dz2bm(fieldz,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldz[i, j+id] - fieldz[i, j-id-1])
    return value

## 2D differential Rotated Staggered Grid ### 
# Forward difference
@ti.func
def Drx2fm(fieldx,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id]*(fieldx[i+id+1,j+id+1] - fieldx[i-id, j-id])
    return value
@ti.func
def Drz2fm(fieldy,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[i-id, j+id+1] - fieldy[i+id+1, j-id])
    return value
# Backward difference
@ti.func
def Drx2bm(fieldx,i,j,coff,n):
    value=0.
    for id in range(n):
        value +=  coff[id]*(fieldx[i+id,j+id] - fieldx[i-id-1, j-id-1])
    return value
@ti.func
def Drz2bm(fieldy,i,j,coff,n):
    value=0.
    for id in range(n):
        value += coff[id] * (fieldy[i-id-1, j+id] - fieldy[i+id, j-id-1])
    return value




