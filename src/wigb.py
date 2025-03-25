import numpy as np
import matplotlib.pyplot as plt

def wigb(a=[],scal=0,x=0,z=0,amx=0):
#  WIGB: Plot seismic data using wiggles.
#
#  WIGB(a,scal,x,z,amx) 
#
#  IN  a:     地震数据 (a ndarray, traces are columns)
#      scale: multiple data by scal
#      x:     x轴(often offset)
#      z:     y轴 (often time)
#
#  Note
#
#    If only 'a' is enter, 'scal,x,z,amn,amx' are set automatically; 
#    otherwise, 'scal' is a scalar; 'x, z' are vectors for annotation in 
#    offset and time, amx are the amplitude range.
#
    if a==[]:
        nx, nz = 10, 10
        a = np.random.random((nz,nx))
        # print(a)
    nz, nx = a.shape

    trmx = np.max(np.abs(a),axis=0)
    if amx==0:
        amx = np.mean(trmx)
    if x==0:
        x = np.arange(1,nx+1,1)
        z = np.arange(1,nz+1,1)
    if scal==0:
        scal = 1
    
    if nx <=1:
        print('ERR:PlotWig: nx has to be more than 1')
        return 
    
# take the average as dx
    dx1 = np.abs(x[1:nx]-x[0:nx-1])
    dx = np.median(dx1)


    dz = z[1]-z[0]
    xmax, xmin = a.max(), a.min()

    a = a * dx / amx
    a = a * scal

    print(' PlotWig: data range [%f, %f], plotted max %f \n'%(xmin,xmax,amx))

    # set display range

    x1 = min(x) - 2.0*dx
    x2 = max(x) - 2.0*dx
    z1 = min(z) - dz
    z2 = max(z) - dz

    fig = plt.figure()
    plt.xlim([x1,x2])
    plt.ylim([z1,z2])
    plt.gca().invert_yaxis()


    zstart, zend = z[0], z[-1]

    fillcolor = [0, 0, 0]
    linecolor = [0, 0, 0]
    linewidth = 1.


    for i in range(0,nx):
        if not trmx[i]==0:
            tr = a[:,i]
            s = np.sign(tr)
            i1 = []
            for j in range(0,nz):
                if j==nz-1:
                    continue
                if not s[j]==s[j+1]:
                    i1.append(j)
            npos = len(i1)

            i1 = np.array(i1)
            if len(i1)==0:
                zadd=np.array(())
            else:
                zadd = i1 + tr[i1] / (tr[i1]-tr[i1+1])
            aadd = np.zeros(zadd.shape)

            zpos = np.where(tr>0)
            tmp = np.append(zpos,zadd)
            zz = np.sort(tmp)
            # iz =np.array(())
            # for j in range(0,len(tmp)):
            #     iz=np.append(iz,np.where(zz==tmp[j]))
            iz = np.argsort(tmp)
            aa = np.append(tr[zpos],aadd)
            iz=iz.astype(int)
            aa = aa[iz]


            if tr[0]>0:
                a0,z0=0,1.00
            else:
                a0,z0=0,zadd[0]
            if tr[nz-1]>0:
                a1,z1=0,nz
            else:
                a1,z1=0,zadd.max()
            
            zz = np.append(np.append(np.append(z0,zz),z1),z0)
            aa = np.append(np.append(np.append(a0,aa),a1),a0)

            zzz = zstart + zz * dz - dz
            plt.fill(aa+x[i], zzz+1, color=fillcolor)
            plt.plot(x[i]+[0,0],[zstart,zend],color=[1,1,1])

            plt.plot(x[i]+tr,z,color=linecolor,linewidth=linewidth)
        else:
            plt.plot([x[i],x[i]],[zstart,zend],color=linecolor,linewidth=linewidth)
    plt.show()


        

if __name__=='__main__':
    wigb()
