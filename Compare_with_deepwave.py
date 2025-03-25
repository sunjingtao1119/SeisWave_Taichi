import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import elastic
import time

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
ny = 512
nx = 512
dx = 1

vp = torch.ones(ny, nx, device=device) * 2500
vs = torch.ones(ny, nx, device=device) * 1500
rho = torch.ones(ny, nx, device=device) * 2200
lamb, mu, buoyancy = (
    deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
)

freq = 60 
nt = 2000
dt = 0.0001
peak_time = 1.5 / freq

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .reshape(1, 1, -1).to(device)
)

# single body forces in y and x dimensions
source_locations = torch.tensor([[[150, 150]]]).to(device)
receiver_locations = torch.tensor([[[200, 200  ]]]).to(device)
# body force in y located at (35.5, 35.5)

ts = time.time()
out = elastic(
    lamb, mu, buoyancy,
    dx, dt,
    source_amplitudes_y=source_amplitudes,
    source_locations_y=source_locations,
    receiver_locations_y=receiver_locations,
    pml_freq=freq,
    accuracy=4,
)
tend = time.time()
print(f'{tend-ts:.3} sec')
data=out[14][0][0]

plt.plot(out[14][0][0])
plt.show()