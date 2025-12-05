import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load and label datasets
# Read entire lines (since some titles contain commas)
def load_titles(filename):
    # Read each line as one record, even if it has commas or quotes
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip newlines and surrounding quotes if present
    titles = [line.strip() for line in lines if line.strip()]

    # Return a DataFrame
    return pd.DataFrame({"title": titles})

# Add labels: 1 = viral, 0 = not viral
viral_df = load_titles("data/viral_title.csv")
viral_df["label"] = 1

no_viral_df = load_titles("data/no_viral_title.csv")
no_viral_df["label"] = 0

# Combine
df = pd.concat([viral_df, no_viral_df], ignore_index=True)
print(f"Dataset loaded: {len(df)} samples")

#Split train/test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

output_dir = "data"
output_file_csv = "train_title.csv"
output_path_csv = os.path.join(output_dir, output_file_csv)
train_df.to_csv(output_path_csv, index=False) 

output_file_csv = "test_title.csv"
output_path_csv = os.path.join(output_dir, output_file_csv)
test_df.to_csv(output_path_csv, index=False) 

print("Dataset successfully saved")


