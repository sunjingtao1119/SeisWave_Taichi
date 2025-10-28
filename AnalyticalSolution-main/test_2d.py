import numpy as np
import matplotlib.pyplot as plt
from AnalyticalSolution import AnalyticalSolution
x = 100   # x receiver coordinate 
y = 100   # y receiver coodinate
z = 100   # z receiver coodinate

rho = 2000            # Density kg/m^3
vp  = 2000            # p-wave velocity
vs  = 1150   # S-wave velocity
M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) # Moment tensor
f0   = 60              # Dominant frequency (Hz)
M0   = 1*1e16             # Scalar Moment 
tmin = 0.0                # Minimum observation time (s)
tmax = 0.2403                # Maximum observation time (s) 
dt   = 0.0003              # Time interval (s)
V = AnalyticalSolution(vp, vs, rho, x, y, z, tmin, tmax, dt, f0, M0, M, dim='2D', comp = 'velocity', verbose = False)
comps = ['Vx', 'Vz']

fig = plt.figure(figsize=(8, 6))
for i, comp in enumerate(comps):
    plt.subplot(len(comps), 1, i+1)
    plt.plot(V['t'], V[comp])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend([comp])
plt.show()