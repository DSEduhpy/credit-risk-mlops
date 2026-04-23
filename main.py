import pandas as pd
import os

# Check if the parquet file exists, if not read from CSV and save as parquet for faster loading next time
if not os.path.exists("data.parquet"):
    df = pd.read_csv("application_train.csv")
    df.to_parquet("data.parquet", index=False)
else:
    df = pd.read_parquet("data.parquet")

print(df.shape)



# Check the distribution of the target variable
print(df['TARGET'].value_counts(normalize=True))

missing = df.isnull().mean().sort_values(ascending=False)
print(missing.head(10))

