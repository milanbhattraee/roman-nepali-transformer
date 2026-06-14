import pandas as pd
import torch
# (Assume you import your Dataset, DataLoader, and Model classes from train.py)
# from train import YourDatasetClass, YourModelClass

print("--- Running Sanity Check ---")

# 1. Load just 10 rows of your clean data
df = pd.read_csv('data/roman_nepali_clean.csv', nrows=10)
print("Testing with samples:")
print(df)

# 2. Initialize your model with small parameters just to test the logic
# (Or use your exact model structure)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 3. Train for 50-100 epochs on JUST these 10 rows.
# Because the dataset is tiny, this will take less than 60 seconds.

# 4. Run an Inference/Prediction test on one of the inputs:
# test_word = "muskuraundai"
# predicted = your_inference_function(test_word)
# print(f"Input: {test_word} -> Predicted: {predicted} (Expected: मुस्कुराउँदै)")