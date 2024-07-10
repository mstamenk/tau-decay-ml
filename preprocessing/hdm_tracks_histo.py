import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('reading in csv files...')
# Read CSV files into DataFrames
df1 = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/maxpt_matched_pion_data.csv')
print('read in first')
df2 = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/pt_nonmatched_pion_data.csv')
print('read in both')
print('now getting randoms')
sample_df1 = df1.sample(n=1000, random_state=42)
print('got first random')
sample_df2 = df2.sample(n=1000, random_state=42)
#print(sample_df2)
print('got second random')
print('now making histos')

df_bin_width = 0.5
df_bins = np.arange(min(sample_df1['pion_max_pt'].values), max(sample_df1['pion_max_pt'].values) + df_bin_width, df_bin_width)

df2_bin_width = 0.5
df2_bins = np.arange(min(sample_df2['unmatched_pion_pt'].values), max(sample_df2['unmatched_pion_pt'].values) + df2_bin_width, df2_bin_width)


plt.hist(sample_df1['pion_max_pt'], bins=df_bins, alpha=0.5, label='Max pT of Matched RECO Pions', color='blue')
print('first histo made')
plt.hist(sample_df2['unmatched_pion_pt'], bins=df2_bins, alpha=0.5, label='All pT of Nonmatched RECO Pions', color='orange')
print('second histo made')

plt.xlabel('pT')
plt.ylabel('Frequency')
plt.title('pT of Matched vs Nonmatched RECO Pions')
print('saving fig...')
plt.show()
print('shoulda showed plot')
plt.savefig('matching_pt_graph.png')
print('done.')