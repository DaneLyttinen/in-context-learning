import polars as pl
import numpy as np

# Create a sample DataFrame
df = pl.DataFrame({
    "A": [1, 2, 3, 4, 5],
    "B": ["one", "two", "three", "four", "five"],
    "C": np.random.rand(5)
})

# Shuffle the DataFrame
shuffled_df = df.shuffle()

# Display the original and shuffled DataFrame
print("Original DataFrame:")
print(df)
print("\nShuffled DataFrame:")
print(shuffled_df)