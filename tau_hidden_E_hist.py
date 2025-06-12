from fast_histogram import histogram1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/afs/cern.ch/user/j/jlippert/public/tau-decay-ml/data/200k_varied_Y_boosted.csv')

pi_mass = 0.13957
tau_mass = 1.7769

pi1_px = data['pi1_pt'] * np.cos(data['pi1_phi'])
pi1_py = data['pi1_pt'] * np.sin(data['pi1_phi'])
pi1_pz = data['pi1_pt'] * np.sinh(data['pi1_eta'])
pi1_E = np.sqrt(pi1_px**2 + pi1_py**2 + pi1_pz**2 + pi_mass**2)
p_1 = [pi1_E, pi1_px, pi1_py, pi1_pz]

pi2_px = data['pi2_pt'] * np.cos(data['pi2_phi'])
pi2_py = data['pi2_pt'] * np.sin(data['pi2_phi'])
pi2_pz = data['pi2_pt'] * np.sinh(data['pi2_eta'])
pi2_E = np.sqrt(pi2_px**2 + pi2_py**2 + pi2_pz**2 + pi_mass**2)
p_2 = [pi2_E, pi2_px, pi2_py, pi2_pz]

pi3_px = data['pi3_pt'] * np.cos(data['pi3_phi'])
pi3_py = data['pi3_pt'] * np.sin(data['pi3_phi'])
pi3_pz = data['pi3_pt'] * np.sinh(data['pi3_eta'])
pi3_E = np.sqrt(pi3_px**2 + pi3_py**2 + pi3_pz**2 + pi_mass**2)
p_3 = [pi3_E, pi3_px, pi3_py, pi3_pz]

E_missing = tau_mass - pi1_E - pi2_E - pi3_E

neu_px = data['neu_pt'] * np.cos(data['neu_phi'])
neu_py = data['neu_pt'] * np.sin(data['neu_phi'])
neu_pz = data['neu_pt'] * np.sinh(data['neu_eta'])
neu_E = np.sqrt(neu_px**2 + neu_py**2 + neu_pz**2)

total_px = pi1_px + pi2_px + pi3_px + neu_px
total_py = pi1_py + pi2_py + pi3_py + neu_py
total_py = pi1_py + pi2_py + pi3_py + neu_py
print("total px: " + total_px)
print("total py: " + total_py)
print("total pz: " + total_pz)

# Parameters for the histogram
bins = 80  # Number of bins
range_min = -0.2
range_max = 2.8

# Compute the histogram
hist1 = histogram1d(E_missing, bins=bins, range=[range_min, range_max])
# hist2 = histogram1d(neu_E, bins=bins, range=[range_min, range_max])

# Create bin edges
bin_edges = np.linspace(range_min, range_max, bins + 1)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist1, width=np.diff(bin_edges), edgecolor='blue')
# plt.bar(bin_edges[:-1], hist2, width=np.diff(bin_edges), edgecolor='green')
plt.title('Missing Energy')
plt.xlabel('GeV')
plt.ylabel('Frequency')
plt.savefig('/afs/cern.ch/user/j/jlippert/public/tau-decay-ml/histograms/missing_energy.png')