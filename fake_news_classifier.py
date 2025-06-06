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

from sklearn.feature_extraction.text import TfidfVectorizer

# Extract features from the 'text' column using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf_vectorizer.fit_transform(data['text'])

# Target labels
y = data['label']

from sklearn.model_selection import train_test_split

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("Model training completed.")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
