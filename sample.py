import pandas as pd
import random

def sample_csv(input_file: str, p: float):
    """
    Reads a CSV file, randomly samples p% of the data, and saves it to a new CSV file.
    
    Args:
    input_file (str): The path to the input CSV file.
    p (float): The percentage of data to sample (between 0 and 100).
    
    Returns:
    None
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Calculate the number of rows to sample based on percentage
    sample_size = int(len(df) * (p / 100))
    
    # Randomly sample the data
    sampled_df = df.sample(n=sample_size, random_state=random.seed(42))
    
    # Create a new file name with the percentage included
    output_file = f"./data/{p}_Percent.csv"
    
    # Save the sampled data to a new CSV file
    sampled_df.to_csv(output_file, index=False)
    print(f"Sampled data saved to {output_file}")

sample_csv('./data/xaa.csv', 10)