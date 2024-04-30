import pandas as pd
import numpy as np
import pickle
# Create a sample DataFrame
res_dict = pd.read_pickle("data/results/transformer_train_38_16_16.pkl")


import polars as pl

# Create two example dataframes
df1 = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

df2 = pl.DataFrame({
    "id": [4, 5],
    "name": ["David", "Eve"]
})

# Concatenate the dataframes vertically
concatenated_df = pl.concat([df1, df2])

# Display the result
print(concatenated_df)

