import taichi as ti
'''
  c is ti.field and used to store the n-point stencil differece coefficient
'''
## 3D differential Lebedev Staggered Grid ###
# Forward difference
@ti.func
def Drfm(fieldx,i,j,k,coff,n):
    value1=0.
    value2=0.
    value3=0.
    value4=0.
    for id in range(n):
        value1 += coff[id] * (fieldx[i+id+1, j+id+1,k+id+1] - fieldx[i-id, j-id,k-id])  
    
    for id in range(n):
        value2 += coff[id] * (fieldx[i+id+1, j-id,k+id+1] - fieldx[i-id, j+id+1,k-id])
    
    for id in range(n):
        value3 += coff[id] * (fieldx[i-id,j+id+1,k+id+1] - fieldx[i+id+1, j-id,k-id])

    for id in range(n):
        value4 += coff[id] * (fieldx[i-id, j-id,k+id+1] - fieldx[i+id+1, j+id+1,k-id])

    return value1,value2,value3,value4

# Backward difference
@ti.func
def Drbm(fieldx,i,j,k,coff,n):
    value1=0.
    value2=0.
    value3=0.
    value4=0.
    for id in range(n):
        value1 += coff[id] * (fieldx[i+id, j+id,k+id] - fieldx[i-id-1, j-id-1,k-id-1])
    for id in range(n):
        value2 += coff[id] *(fieldx[i+id, j-id-1,k+id] - fieldx[i-id-1, j+id,k-id-1])

    for id in range(n):
        value3 += coff[id] * (fieldx[i-id-1,j+id,k+id] - fieldx[i+id, j-id-1,k-id-1])
    for id in range(n):
        value4 += coff[id] * (fieldx[i-id-1, j-id-1,k+id] - fieldx[i+id, j+id,k-id-1])

    return value1,value2,value3,value4



