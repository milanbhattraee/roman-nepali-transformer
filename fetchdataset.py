import pandas as pd
from datasets import load_dataset

def fetch_and_save_dataset():
    print("Connecting to Hugging Face...")
    print("Downloading 'Saugatkafley/Nepali-Roman-Transliteration' (this may take a few minutes)...")
    
    # 1. Load the dataset from Hugging Face
    dataset = load_dataset("Saugatkafley/Nepali-Roman-Transliteration")
    
    # 2. Select the 'train' split which contains the main data
    train_data = dataset['train']
    total_rows = len(train_data)
    print(f"Download complete! Successfully loaded {total_rows:,} rows.")
    
    # 3. Convert the Hugging Face dataset to a Pandas DataFrame
    print("Converting to tabular format...")
    df = train_data.to_pandas()
    
    # Optional: If you want to train a smaller model first to test your architecture, 
    # you can uncomment the line below to only save the first 100,000 rows.
    # df = df.head(100000)

    # 4. Save the DataFrame to a CSV file
    output_filename = "ready_to_train_nepali.csv"
    print(f"Saving data to '{output_filename}'...")
    
    # using utf-8 encoding is crucial for Devanagari script
    df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print("Done! Your CSV file is fully prepared for AI training.")

if __name__ == "__main__":
    fetch_and_save_dataset()