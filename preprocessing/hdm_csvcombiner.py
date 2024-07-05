import pandas as pd
import os

# Directory containing the CSV files
csv_directory = '/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing'

# List to hold each dataframe
dataframes_gen = []
dataframes_reco = []

# Loop through all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        if filename.startswith('gen'):
            file_path = os.path.join(csv_directory, filename)
        
            # Read the CSV file
            df_gen = pd.read_csv(file_path)
        
            # Append the dataframe to the list
            dataframes_gen.append(df_gen)

        if filename.startswith('reco'):
            file_path = os.path.join(csv_directory, filename)
        
            # Read the CSV file
            df_reco = pd.read_csv(file_path)
        
            # Append the dataframe to the list
            dataframes_reco.append(df_reco)

# Concatenate all dataframes in the list
combined_df_gen = pd.concat(dataframes_gen, ignore_index=True)
combined_df_reco = pd.concat(dataframes_reco, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df_gen.to_csv('gen_alldata.csv', index=False)
combined_df_reco.to_csv('reco_alldata.csv', index=False)

print("All reco CSV files have been combined into 'reco_alldata.csv' and all gen CSV files have been combined into 'gen_alldata.csv")