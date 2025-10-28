
 # reference matlab code   optimal_FD  Written by Erik Koene  
 # https://github.com/efmkoene/optimal_FD/tree/master  

 
import numpy as np

def FD_liu(L,error):
    """
    Compute least-squares optimal finite-difference coefficients.

    Parameters
    ----------
    L : int
        Number of coefficients.
    order : int
        Derivative order (1 or 2).
    error : float
        Maximum wavenumber error.

    Returns
    -------
    c : ndarray
        Optimal FD coefficients.
    max_wavenumber : float
        Maximum wavenumber within the desired error range.
    """
    # The optimization functions
    # The computed wavenumber
    def cfn(m, n, b):
        return np.sum(
                4 * (np.sin((m + 0.5) * b) - 2 * (m + 0.5) * np.sin(b / 2)) *
                (np.sin((n + 0.5) * b) - 2 * (n + 0.5) * np.sin(b / 2))
            ) / len(b)

    # The desired wavenumber
    def dfn(n, b):
        return np.sum(
                2 * (np.sin((n + 0.5) * b) - 2 * (n + 0.5) * np.sin(b / 2)) *
                (b - 2 * np.sin(b / 2))
            ) / len(b)

    def Dxj(k, d, L):
            temp1=np.arange(1, L + 1)
            temp1=temp1.reshape((L,1))  
            d=d.reshape((L,1))
            data= 2 * d * (temp1 - 0.5)* np.cos((temp1 - 0.5) * k)
            return np.max(
                np.real(np.sum(data, axis=0)) - 1
            )

    #Testing the error for different wavenumber ranges
    db = 0.001
    wavenumbers = np.arange(db, np.pi + db, db)
    wavenumerror = []
    C = np.zeros((L-1, L-1))
    d = np.zeros(L-1)
    for bmax in wavenumbers:
        b = np.arange(0, bmax + db, db) 
        for m in range(L-1):
            m0=m+1
            for n in range(L-1):
                n0=n+1
                C[n,m] = cfn(m0, n0, b)
            d[m] = dfn(m0, b)
        # Least squares solution
        x = np.linalg.lstsq(C, d, rcond=None)[0]
        c = np.zeros(L)
        c[0] = 1 - np.dot(np.arange(3, 2 * L, 2), x)
        c[1:L] = x
        wavenumerror.append(Dxj(b, c, L))

    wavenumerror = np.array(wavenumerror)
    #Find the maximum wavenumber within the error bound
    mask = np.abs(wavenumerror) < error
    if not np.any(mask):
        raise RuntimeError("No wavenumber found within the error bound.")
    tmp = wavenumbers[mask]

    # System of equations for the desired wavenumber range
    b = np.arange(0, tmp[-1] + db, db)
    for m in range(L - 1):
        m0=m+1
        for n in range(L - 1):
            n0=n+1
            C[n, m] = cfn(m0, n0, b)
        d[m] = dfn(m0, b)
   # Find least squares solution for desired wavenumber range
    x = np.linalg.lstsq(C, d, rcond=None)[0]
   # Generate list of coefficients
    c = np.zeros(L)
    c[0] = 1 - np.dot(np.arange(3, 2 * L, 2), x)
    c[1:L] = x

    return c,tmp[-1]