import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# constants
M_PI = 0.13957  # GeV (mass of a charged pion)
M_NU = 0.0      # GeV (mass of neutrino approximated as zero)

# function to convert (pT, eta, phi, M) to a 4-momentum vector
def four_vector(pT, eta, phi, M):
    px = pT * np.cos(phi)
    py = pT * np.sin(phi)
    pz = pT * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + M**2)
    return np.array([E, px, py, pz])

# loading CSV file
csv_path = "/afs/cern.ch/user/z/zfang/public/tau-decay-ml/data/200k_varied_Y_boosted.csv"
df = pd.read_csv(csv_path)

# computing invariant masses
invariant_masses = []

for _, row in df.iterrows():
    pi1 = four_vector(row['pi1_pt'], row['pi1_eta'], row['pi1_phi'], M_PI)
    pi2 = four_vector(row['pi2_pt'], row['pi2_eta'], row['pi2_phi'], M_PI)
    pi3 = four_vector(row['pi3_pt'], row['pi3_eta'], row['pi3_phi'], M_PI)
    neu = four_vector(row['neu_pt'],  row['neu_eta'],  row['neu_phi'],  M_NU)
    
    total = pi1 + pi2 + pi3 + neu 
    E, px, py, pz = total
    m_tau = np.sqrt(np.maximum(E**2 - (px**2 + py**2 + pz**2), 0))  # prevent sqrt of negative
    invariant_masses.append(m_tau)

# plotting histogram
plt.figure(figsize=(10, 6))
plt.hist(invariant_masses, bins=50, edgecolor='black')
plt.xlabel("Invariant Mass of Tau (GeV)")
plt.ylabel("Number of Events")
plt.title("Invariant Mass Distribution of Tau Leptons")
plt.grid(True)
plt.savefig("/afs/cern.ch/user/z/zfang/public/tau-decay-ml/histograms/tau_invariant_mass_hist.png")
plt.close()