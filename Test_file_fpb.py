from datasets import load_dataset
ds = load_dataset("descartes100/enhanced-financial-phrasebank")
fpb = ds["train"].to_pandas()
print(f"Total rows: {len(fpb)}")
print(f"Columns: {fpb.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(fpb.head(3))
print(f"\nLabel value counts:")
print(fpb.iloc[:, -1].value_counts())  # whatever the label column is called