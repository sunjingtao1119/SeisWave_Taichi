
import numpy as np
import matplotlib.pyplot as plt

from ReadSegy import read_segy
import time

#
# read model
filename_rho = './model/MODEL_DENSITY_1.25m.segy'
filename_vp='./model/MODEL_P-WAVE_VELOCITY_1.25m.segy'
filename_vs='./model/MODEL_S-WAVE_VELOCITY_1.25m.segy'

# Geometry model parameters
rho_np = read_segy(filename_rho,shotnum=1)
vp_np  =read_segy(filename_vp,shotnum=1)
vs_np  =read_segy(filename_vs,shotnum=1)
rho_np=rho_np[::10,::20]
vp_np=vp_np[::10,::20]
vs_np=vs_np[::10,::20]
print(rho_np.shape)
plt.imshow(vp_np,cmap='seismic')
plt.show()
