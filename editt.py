import pandas as pd

# Read the CSV file
df = pd.read_csv('data.csv')

# Option 1: Most common & recommended way
# Replace empty string, whitespace-only, and NaN with None (null in output)
df['suggestion'] = df['suggestion'].replace(['', ' ', '  ', '   ', '\t', '\n'], None)

# OR more thorough version (catches all kinds of whitespace):
# df['suggestion'] = df['suggestion'].replace(r'^\s*$', None, regex=True)

# Option 2: Alternative - convert empty-looking values to None
# df['suggestion'] = df['suggestion'].where(df['suggestion'].str.strip().ne(''), None)

# Optional: Check how many were changed
print("Number of empty suggestions replaced:", df['suggestion'].isna().sum())

# Save back to the same file (or choose a different name)
df.to_csv('data.csv', index=False, encoding='utf-8')

print("Done! Empty cells in 'suggestion' column have been set to null.")