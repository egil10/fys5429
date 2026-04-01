import pandas as pd
from pathlib import Path

def generate_samples():
    # Define the directory containing the generated data
    data_dir = Path("../data/generated")
    
    # Find all parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No .parquet files found in {data_dir.resolve()}")
        return

    for p_file in parquet_files:
        # Load the full parquet dataset
        df = pd.read_parquet(p_file)
        
        sample_size = min(100, len(df))
        
        # If 'point_type' exists, doing a stratified sample gives a nice mixed preview
        if 'point_type' in df.columns:
            # Calculate how many points per type we need
            n_types = df['point_type'].nunique()
            points_per_type = max(1, sample_size // n_types)
            df_sample = df.groupby('point_type').sample(n=points_per_type, replace=True).head(sample_size)
        else:
            # Simple random sample
            df_sample = df.sample(n=sample_size, random_state=42)
            
        # Define the output CSV name
        out_csv = data_dir / f"{p_file.stem}_sample.csv"
        
        # Save to CSV
        df_sample.to_csv(out_csv, index=False)
        print(f"Generated sample: {out_csv.name} ({len(df_sample)} rows)")

if __name__ == "__main__":
    generate_samples()
