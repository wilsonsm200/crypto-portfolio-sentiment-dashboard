import pandas as pd
import os

INPUT_FILE = './data/factors/factor_matrix.csv'
OUTPUT_DIR = './data/factors/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'factor_matrix_aligned.csv')

# Load factor matrix
df = pd.read_csv(INPUT_FILE, parse_dates=['time'])

print(f"[+] Loaded factor matrix shape: {df.shape}")

# Fill missing values rather than dropping to avoid losing rows
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

print(f"[✓] After filling missing values: {df.shape}")

# Save aligned data
df.to_csv(OUTPUT_FILE, index=False)

print(f"[💾] Aligned factor matrix saved: {OUTPUT_FILE}")
print(df.head())
