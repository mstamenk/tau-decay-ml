from fast_histogram import histogram1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

gen_data = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/gen_info_both_proper_decay.csv')


# Parameters for the histogram
bins = 80  # Number of bins
range_min = 8
range_max = 16



# Compute the histogram
hist1 = histogram1d(gen_data['gen_upsilon_mass'], bins=bins, range=[range_min, range_max])
# Create bin edges
bin_edges = np.linspace(range_min, range_max, bins + 1)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist1, width=np.diff(bin_edges), edgecolor='blue')
plt.title('Invariant Mass of Upsilon')
plt.xlabel('Invariant Mass (GeV)')
plt.ylabel('Frequency (of 9000 Upsilons)')
plt.savefig('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/gen_upsilon_mass_hist.png')



