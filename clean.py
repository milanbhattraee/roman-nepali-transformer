import pandas as pd

# 1. Load your dataset (assuming it's saved as 'nepali_data.csv')
# If you don't have pandas installed, run: pip install pandas
df = pd.read_csv('ready_to_train_nepali.csv')

# 2. Drop the 'unique_identifier' column completely
df = df.drop(columns=['unique_identifier'])

# 3. Rename columns for machine learning clarity
df = df.rename(columns={
    'native word': 'devanagari_target',
    'english word': 'roman_input'
})

# 4. (Optional but recommended) Convert the roman_input to lowercase 
# This prevents the model from treating 'Muskuraundai' and 'muskuraundai' differently
df['roman_input'] = df['roman_input'].str.lower()

# 5. Save the cleaned dataset to a new file ready for training
df.to_csv('cleaned_training_data.csv', index=False, encoding='utf-8')

print("Data cleaned successfully! Here is a preview:")
print(df.head())