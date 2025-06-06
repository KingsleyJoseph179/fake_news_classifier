import pandas as pd

# Load the datasets
fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')

# Show the first 5 rows of each dataset
print("Fake news sample:")
print(fake_news.head())

print("\nReal news sample:")
print(real_news.head())

# Check the shape (rows, columns)
print(f"\nFake news dataset shape: {fake_news.shape}")
print(f"Real news dataset shape: {real_news.shape}")

# Add a new column 'label' for classification: 0 for fake, 1 for real
fake_news['label'] = 0
real_news['label'] = 1

# Combine both datasets into one DataFrame
data = pd.concat([fake_news, real_news], ignore_index=True)

# Shuffle the combined data to mix fake and real news
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Show combined dataset info and first 5 rows
print(f"\nCombined dataset shape: {data.shape}")
print(data.head())
