import pandas as pd
import numpy as np

# Read the CSV file
file_path = "/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/LabelMix/evaluation/data/processed/vitdet_metrics_aggregated.csv"
df = pd.read_csv(file_path)

# Remove the 'ptseeds' column
if 'ptseeds' in df.columns:
    df = df.drop(columns=['ptseeds'])

# Identify numeric columns (excluding the first two columns which are model and checkpoint-type)
numeric_columns = df.columns[2:]  # Skip the first two categorical columns

# Round all numeric values to 2 decimal places
for col in numeric_columns:
    df[col] = df[col].round(2)

# Save the processed CSV back to the same file
df.to_csv(file_path, index=False)

print("CSV processing completed:")
print(f"- Removed 'ptseeds' column")
print(f"- Rounded all numeric values to 2 decimal places")
print(f"- Processed file saved to: {file_path}")