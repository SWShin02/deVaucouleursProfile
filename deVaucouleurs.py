import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

galaxy_name = 'M60'
galaxy = pd.read_csv(
    filepath_or_buffer=f'data/{galaxy_name}.tbl', 
    skiprows=11, 
    sep=r'\s+', 
    names=['Index', 'Lon_LeftReadout', 'Lat_LeftReadout', 'Lon_RightReadout', 'Lat_RightReadout', 'Distance','Flux'],
    index_col=0
)
distance = galaxy['Distance'].to_numpy()
flux = galaxy['Flux'].to_numpy()
flux = np.flip(flux) # 데이터가 뒤집혀있는 경우

# Masking
mask = distance>50
distance_sorted= distance[mask]
flux_sorted = flux[mask]

r_inner = distance[:-1]  # Inner radii of annuli
r_outer = distance[1:]   # Outer radii of annuli

areas = np.pi * (r_outer**2 - r_inner**2)  # Areas of annuli

d_flux = flux[-1]*areas

# Compute cumulative flux
cumulative_flux = np.cumsum(d_flux)

total_flux = cumulative_flux[-1]

# Find the radius where cumulative flux is half of the total flux
half_flux = total_flux / 2
r_e_index = np.searchsorted(cumulative_flux, half_flux)
r_e = distance[r_e_index]


# Compute surface brightness μ(r) = -2.5 * log10(Flux)
# Since Flux_sorted is proportional to surface brightness
mu_r = -2.5 * np.log10(flux/flux[r_e_index-5:r_e_index+5].mean())
mu_r_sorted = mu_r[mask]

# Define the generalized de Vaucouleurs profile function
def deVaucouleurs_profile(r, n):
    return 8.3268 * ((r / r_e) ** (1 / n) - 1)

params, _ = curve_fit(
    f=deVaucouleurs_profile, 
    xdata= distance_sorted,
    ydata= mu_r_sorted,
    p0= 4,
    bounds=((1), (np.inf))
)
n = params[0]
distance_sample = np.linspace(distance[0], distance[-1], 1000)
mu_fit = deVaucouleurs_profile(distance_sample, n)
mu_4 = deVaucouleurs_profile(distance_sample, 4)

print(r_e)

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.scatter(distance, mu_r, s=10)
ax.plot(distance_sample, mu_fit, c='gray', label=f'n={n:.1f}')
ax.plot(distance_sample, mu_4, c='gray', ls='--', label='n=4')


# Details
ax.set_xlim((0, distance.max())); ax.set_ylim((-1.5,0.5))
ax.legend(edgecolor='none')
ax.tick_params(direction='in')
ax.invert_yaxis()  # Invert y-axis because lower magnitudes are brighter
ax.grid(True)

ax.set_title(galaxy_name)
ax.set_xlabel('Distance [arcsec]')
ax.set_ylabel(r'$\mu_{r}-\mu_{e}$ [mag/arcsec$^2$]')

# Save
fig.savefig(f'figures/{galaxy_name}_mu.png')