import pandas as pd
import numpy as np

# Constants
M_PI = 0.13957  # GeV
M_NU = 0.0      # Approx. massless

# Load the data
df_gen = pd.read_csv("/home/zfang31/tau-decay-ml/data/gen_alldata.csv")
df_reco = pd.read_csv("/home/zfang31/tau-decay-ml/data/reco_alldata.csv")
assert len(df_gen) == len(df_reco), "Mismatch in number of events"

# Four-vector computation
def four_vector(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E  = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([E, px, py, pz], axis=-1)

# Lorentz boost function
def boost_to_rest_frame(p, boost_vec):
    b2 = np.sum(boost_vec**2, axis=-1)
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(p[:, 1:] * boost_vec, axis=-1)
    gamma2 = (gamma - 1.0) / (b2 + 1e-8)

    # Reshape for broadcasting
    gamma = gamma[:, np.newaxis]
    bp = bp[:, np.newaxis]
    gamma2 = gamma2[:, np.newaxis]
    factor = gamma2 * bp + gamma * p[:, [0]]

    boosted_p = np.empty_like(p)
    boosted_p[:, 0] = gamma[:, 0] * p[:, 0] - bp[:, 0]
    boosted_p[:, 1:] = p[:, 1:] + factor * boost_vec
    return boosted_p

# Convert 4-vector to (pt, eta, phi)
def vec_to_pt_eta_phi(pvec):
    px, py, pz = pvec[:, 1], pvec[:, 2], pvec[:, 3]
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    eta = 0.5 * np.log((p + pz) / (p - pz + 1e-8))  # avoid div by 0
    phi = np.arctan2(py, px)
    return pt, eta, phi

# Reconstruct 4-vectors
pi1 = four_vector(df_gen['pi1_pt'], df_gen['pi1_eta'], df_gen['pi1_phi'], M_PI)
pi2 = four_vector(df_gen['pi2_pt'], df_gen['pi2_eta'], df_gen['pi2_phi'], M_PI)
pi3 = four_vector(df_gen['pi3_pt'], df_gen['pi3_eta'], df_gen['pi3_phi'], M_PI)
neu = four_vector(df_reco['neu_pt'], df_reco['neu_eta'], df_reco['neu_phi'], M_NU)

# Tau 4-momentum = sum
tau = pi1 + pi2 + pi3 + neu
tau_beta = tau[:, 1:] / tau[:, 0:1]  # v/c = p/E

# Boost all particles
n = len(tau)
pions_lab = np.stack([pi1, pi2, pi3], axis=1).reshape(-1, 4)
boost_vecs = np.repeat(-tau_beta, 3, axis=0)
pions_rest = boost_to_rest_frame(pions_lab, boost_vecs).reshape(n, 3, 4)
neu_rest = boost_to_rest_frame(neu, -tau_beta)

# Convert to pt/eta/phi
pt1, eta1, phi1 = vec_to_pt_eta_phi(pions_rest[:, 0])
pt2, eta2, phi2 = vec_to_pt_eta_phi(pions_rest[:, 1])
pt3, eta3, phi3 = vec_to_pt_eta_phi(pions_rest[:, 2])
ptn, etan, phin = vec_to_pt_eta_phi(neu_rest)

# Output
df_out = pd.DataFrame({
    'pi1_pt': pt1, 'pi1_eta': eta1, 'pi1_phi': phi1,
    'pi2_pt': pt2, 'pi2_eta': eta2, 'pi2_phi': phi2,
    'pi3_pt': pt3, 'pi3_eta': eta3, 'pi3_phi': phi3,
    'neu_pt': ptn, 'neu_eta': etan, 'neu_phi': phin,
})

df_out.to_csv("data/boosted_alldata.csv", index=False)
print("Saved boosted pt/eta/phi to 'data/boosted_alldata.csv'")
