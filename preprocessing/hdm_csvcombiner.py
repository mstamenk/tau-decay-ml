import pandas as pd
import os

# Directory containing the CSV files
csv_directory = '/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV'

# List to hold each dataframe
dataframes_gen = []
dataframes_reco = []
dataframes_tracking = []
dataframes_unmatched = []

# Loop through all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        if filename.startswith('tracking'):
            file_path = os.path.join(csv_directory, filename)
            df_tracking = pd.read_csv(file_path)
            dataframes_tracking.append(df_tracking)
        if filename.startswith('tau'):
            file_path = os.path.join(csv_directory, filename)
            df_init_tau = pd.read_csv(file_path)
            df_tau = df_init_tau.rename(columns={'pi1_from_tau_pt': 'pi1_pt', 'pi1_from_tau_eta': 'pi1_eta', 'pi1_from_tau_phi': 'pi1_phi', 'pi2_from_tau_pt': 'pi2_pt', 'pi2_from_tau_eta': 'pi2_eta', 'pi2_from_tau_phi': 'pi2_phi', 'pi3_from_tau_pt': 'pi3_pt', 'pi3_from_tau_eta': 'pi3_eta', 'pi3_from_tau_phi': 'pi3_phi', 'neutrino_from_tau_pt': 'neu_pt', 'neutrino_from_tau_eta': 'neu_eta', 'neutrino_from_tau_phi': 'neu_phi', 'tau_no_neutrino_mass': 'tau_noneu_m', 'tau_with_neutrino_mass': 'tau_m'})
            dataframes_reco.append(df_tau)
        if filename.startswith('anti'):
            file_path = os.path.join(csv_directory, filename)
            df_init_anti = pd.read_csv(file_path)
            df_anti = df_init_anti.rename(columns={'pi1_from_antitau_pt': 'pi1_pt', 'pi1_from_antitau_eta': 'pi1_eta', 'pi1_from_antitau_phi': 'pi1_phi', 'pi2_from_antitau_pt': 'pi2_pt', 'pi2_from_antitau_eta': 'pi2_eta', 'pi2_from_antitau_phi': 'pi2_phi', 'pi3_from_antitau_pt': 'pi3_pt', 'pi3_from_antitau_eta': 'pi3_eta', 'pi3_from_antitau_phi': 'pi3_phi', 'neutrino_from_antitau_pt': 'neu_pt', 'neutrino_from_antitau_eta': 'neu_eta', 'neutrino_from_antitau_phi': 'neu_phi', 'tau_no_neutrino_mass': 'tau_noneu_m', 'tau_with_neutrino_mass': 'tau_m'})
            dataframes_reco.append(df_anti)
        if filename.startswith('both'):
            file_path = os.path.join(csv_directory, filename)
            df_init_both = pd.read_csv(file_path)
            columns_tau = ['pi1_from_tau_pt', 'pi1_from_tau_eta', 'pi1_from_tau_phi', 'pi2_from_tau_pt', 'pi2_from_tau_eta', 'pi2_from_tau_phi', 'pi3_from_tau_pt', 'pi3_from_tau_eta', 'pi3_from_tau_phi',
                           'neutrino_from_tau_pt', 'neutrino_from_tau_eta', 'neutrino_from_tau_phi',
                           'tau_no_neutrino_mass', 'tau_with_neutrino_mass',
                           'upsilon_no_neutrino_mass', 'upsilon_with_neutrino_mass']
            columns_anti = ['pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi', 'pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi', 'pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi',
                           'neutrino_from_antitau_pt', 'neutrino_from_antitau_eta', 'neutrino_from_antitau_phi',
                           'antitau_no_neutrino_mass', 'antitau_with_neutrino_mass',
                           'upsilon_no_neutrino_mass', 'upsilon_with_neutrino_mass']
            df_init_btau = df_init_both.filter(columns_tau)
            df_init_banti = df_init_both.filter(columns_anti)
            df_btau = df_init_btau.rename(columns={'pi1_from_tau_pt': 'pi1_pt', 'pi1_from_tau_eta': 'pi1_eta', 'pi1_from_tau_phi': 'pi1_phi', 'pi2_from_tau_pt': 'pi2_pt', 'pi2_from_tau_eta': 'pi2_eta', 'pi2_from_tau_phi': 'pi2_phi', 'pi3_from_tau_pt': 'pi3_pt', 'pi3_from_tau_eta': 'pi3_eta', 'pi3_from_tau_phi': 'pi3_phi', 'neutrino_from_tau_pt': 'neu_pt', 'neutrino_from_tau_eta': 'neu_eta', 'neutrino_from_tau_phi': 'neu_phi', 'tau_no_neutrino_mass': 'tau_noneu_m', 'tau_with_neutrino_mass': 'tau_m', 'upsilon_no_neutrino_mass': 'ups_noneu_m', 'upsilon_with_neutrino_mass': 'ups_m'})
            df_banti = df_init_banti.rename(columns={'pi1_from_antitau_pt': 'pi1_pt', 'pi1_from_antitau_eta': 'pi1_eta', 'pi1_from_antitau_phi': 'pi1_phi', 'pi2_from_antitau_pt': 'pi2_pt', 'pi2_from_antitau_eta': 'pi2_eta', 'pi2_from_antitau_phi': 'pi2_phi', 'pi3_from_antitau_pt': 'pi3_pt', 'pi3_from_antitau_eta': 'pi3_eta', 'pi3_from_antitau_phi': 'pi3_phi', 'neutrino_from_antitau_pt': 'neu_pt', 'neutrino_from_antitau_eta': 'neu_eta', 'neutrino_from_antitau_phi': 'neu_phi', 'antitau_no_neutrino_mass': 'tau_noneu_m', 'antitau_with_neutrino_mass': 'tau_m', 'upsilon_no_neutrino_mass': 'ups_noneu_m', 'upsilon_with_neutrino_mass': 'ups_m'})
            dataframes_reco.append(df_btau)
            dataframes_reco.append(df_banti)
        if filename.startswith('gen_info_tau'):
            file_path = os.path.join(csv_directory, filename)
            df_init_gentau = pd.read_csv(file_path)
            df_gentau = df_init_gentau.rename(columns={'gen_pi1_from_tau_pt': 'pi1_pt', 'gen_pi1_from_tau_eta': 'pi1_eta', 'gen_pi1_from_tau_phi': 'pi1_phi', 'gen_pi2_from_tau_pt': 'pi2_pt', 'gen_pi2_from_tau_eta': 'pi2_eta', 'gen_pi2_from_tau_phi': 'pi2_phi', 'gen_pi3_from_tau_pt': 'pi3_pt', 'gen_pi3_from_tau_eta': 'pi3_eta', 'gen_pi3_from_tau_phi': 'pi3_phi', 'gen_neutrino_from_tau_pt': 'neu_pt', 'gen_neutrino_from_tau_eta': 'neu_eta', 'gen_neutrino_from_tau_phi': 'neu_phi', 'gen_tau_no_neutrino_mass': 'tau_noneu_m', 'gen_tau_with_neutrino_mass': 'tau_m'})
            dataframes_gen.append(df_gentau)
        if filename.startswith('gen_info_antitau'):
            file_path = os.path.join(csv_directory, filename)
            df_init_genanti = pd.read_csv(file_path)
            df_genanti = df_init_genanti.rename(columns={'gen_pi1_from_antitau_pt': 'pi1_pt', 'gen_pi1_from_antitau_eta': 'pi1_eta', 'gen_pi1_from_antitau_phi': 'pi1_phi', 'gen_pi2_from_antitau_pt': 'pi2_pt', 'gen_pi2_from_antitau_eta': 'pi2_eta', 'gen_pi2_from_antitau_phi': 'pi2_phi', 'gen_pi3_from_antitau_pt': 'pi3_pt', 'gen_pi3_from_antitau_eta': 'pi3_eta', 'gen_pi3_from_antitau_phi': 'pi3_phi', 'gen_neutrino_from_antitau_pt': 'neu_pt', 'gen_neutrino_from_antitau_eta': 'neu_eta', 'gen_neutrino_from_antitau_phi': 'neu_phi', 'gen_tau_no_neutrino_mass': 'tau_noneu_m', 'gen_tau_with_neutrino_mass': 'tau_m'})
            dataframes_gen.append(df_genanti)
        if filename.startswith('gen_info_both'):
            file_path = os.path.join(csv_directory, filename)
            df_init_genboth = pd.read_csv(file_path)
            columns_gentau = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi',
                           'gen_neutrino_from_tau_pt', 'gen_neutrino_from_tau_eta', 'gen_neutrino_from_tau_phi',
                           'gen_tau_no_neutrino_mass', 'gen_tau_with_neutrino_mass',
                           'gen_upsilon_no_neutrino_mass', 'gen_upsilon_with_neutrino_mass']
            columns_genanti = ['gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi',
                           'gen_neutrino_from_antitau_pt', 'gen_neutrino_from_antitau_eta', 'gen_neutrino_from_antitau_phi',
                           'gen_antitau_no_neutrino_mass', 'gen_antitau_with_neutrino_mass',
                           'gen_upsilon_no_neutrino_mass', 'gen_upsilon_with_neutrino_mass']
            df_init_genbtau = df_init_genboth.filter(columns_gentau)
            df_init_genbanti = df_init_genboth.filter(columns_genanti)
            df_genbtau = df_init_genbtau.rename(columns={'gen_pi1_from_tau_pt': 'pi1_pt', 'gen_pi1_from_tau_eta': 'pi1_eta', 'gen_pi1_from_tau_phi': 'pi1_phi', 'gen_pi2_from_tau_pt': 'pi2_pt', 'gen_pi2_from_tau_eta': 'pi2_eta', 'gen_pi2_from_tau_phi': 'pi2_phi', 'gen_pi3_from_tau_pt': 'pi3_pt', 'gen_pi3_from_tau_eta': 'pi3_eta', 'gen_pi3_from_tau_phi': 'pi3_phi', 'gen_neutrino_from_tau_pt': 'neu_pt', 'gen_neutrino_from_tau_eta': 'neu_eta', 'gen_neutrino_from_tau_phi': 'neu_phi', 'gen_tau_no_neutrino_mass': 'tau_noneu_m', 'gen_tau_with_neutrino_mass': 'tau_m', 'gen_upsilon_no_neutrino_mass': 'ups_noneu_m', 'gen_upsilon_with_neutrino_mass': 'ups_m'})
            df_genbanti = df_init_genbanti.rename(columns={'gen_pi1_from_antitau_pt': 'pi1_pt', 'gen_pi1_from_antitau_eta': 'pi1_eta', 'gen_pi1_from_antitau_phi': 'pi1_phi', 'gen_pi2_from_antitau_pt': 'pi2_pt', 'gen_pi2_from_antitau_eta': 'pi2_eta', 'gen_pi2_from_antitau_phi': 'pi2_phi', 'gen_pi3_from_antitau_pt': 'pi3_pt', 'gen_pi3_from_antitau_eta': 'pi3_eta', 'gen_pi3_from_antitau_phi': 'pi3_phi', 'gen_neutrino_from_antitau_pt': 'neu_pt', 'gen_neutrino_from_antitau_eta': 'neu_eta', 'gen_neutrino_from_antitau_phi': 'neu_phi', 'gen_antitau_no_neutrino_mass': 'tau_noneu_m', 'gen_antitau_with_neutrino_mass': 'tau_m', 'gen_upsilon_no_neutrino_mass': 'ups_noneu_m', 'gen_upsilon_with_neutrino_mass': 'ups_m'})
            dataframes_gen.append(df_genbtau)
            dataframes_gen.append(df_genbanti)
        if filename.startswith('unmatched_gen_info_tau'):
            file_path = os.path.join(csv_directory, filename)
            df_init_unmatched_gentau = pd.read_csv(file_path)
            df_unmatched_gentau = df_init_unmatched_gentau.rename(columns={'gen_pi1_from_tau_pt': 'pi1_pt', 'gen_pi1_from_tau_eta': 'pi1_eta', 'gen_pi1_from_tau_phi': 'pi1_phi', 'gen_pi2_from_tau_pt': 'pi2_pt', 'gen_pi2_from_tau_eta': 'pi2_eta', 'gen_pi2_from_tau_phi': 'pi2_phi', 'gen_pi3_from_tau_pt': 'pi3_pt', 'gen_pi3_from_tau_eta': 'pi3_eta', 'gen_pi3_from_tau_phi': 'pi3_phi', 'gen_neutrino_from_tau_pt': 'neu_pt', 'gen_neutrino_from_tau_eta': 'neu_eta', 'gen_neutrino_from_tau_phi': 'neu_phi', 'gen_tau_no_neutrino_mass': 'tau_noneu_m', 'gen_tau_with_neutrino_mass': 'tau_m'})
            dataframes_unmatched.append(df_unmatched_gentau)
        if filename.startswith('unmatched_gen_info_antitau'):
            file_path = os.path.join(csv_directory, filename)
            df_init_unmatched_genanti = pd.read_csv(file_path)
            df_unmatched_genanti = df_init_unmatched_genanti.rename(columns={'gen_pi1_from_antitau_pt': 'pi1_pt', 'gen_pi1_from_antitau_eta': 'pi1_eta', 'gen_pi1_from_antitau_phi': 'pi1_phi', 'gen_pi2_from_antitau_pt': 'pi2_pt', 'gen_pi2_from_antitau_eta': 'pi2_eta', 'gen_pi2_from_antitau_phi': 'pi2_phi', 'gen_pi3_from_antitau_pt': 'pi3_pt', 'gen_pi3_from_antitau_eta': 'pi3_eta', 'gen_pi3_from_antitau_phi': 'pi3_phi', 'gen_neutrino_from_antitau_pt': 'neu_pt', 'gen_neutrino_from_antitau_eta': 'neu_eta', 'gen_neutrino_from_antitau_phi': 'neu_phi', 'gen_tau_no_neutrino_mass': 'tau_noneu_m', 'gen_tau_with_neutrino_mass': 'tau_m'})
            dataframes_unmatched.append(df_unmatched_genanti)



# Concatenate all dataframes in the list
combined_df_tracking = pd.concat(dataframes_tracking, ignore_index=True)
combined_df_reco = pd.concat(dataframes_reco, ignore_index=True)
combined_df_gen = pd.concat(dataframes_gen, ignore_index=True)
combined_df_unmatched = pd.concat(dataframes_unmatched, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df_tracking.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/tracking_frequency_alldata15GeV.csv', index=False)
combined_df_reco.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/reco_alldata15GeV.csv', index=False)
combined_df_gen.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/gen_alldata15GeV.csv', index=False)
combined_df_unmatched.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/unmatched_alldata15GeV.csv', index=False)

print("All CSV files combined")