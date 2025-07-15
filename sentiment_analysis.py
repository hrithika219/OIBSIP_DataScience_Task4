# sentiment_analysis.py

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Step 2: Load dataset
df = pd.read_csv('Twitter_Data.csv')

print("First 5 rows of the data:")
print(df.head())

# Step 3: Drop rows with missing category
df = df.dropna(subset=['category'])

# Step 4: Map numeric sentiment labels to text labels
df['category'] = df['category'].astype(str)
df['category'] = df['category'].map({
    '-1.0': 'Negative',
    '0.0': 'Neutral',
    '1.0': 'Positive'
})

# Step 5: Download stopwords
nltk.download('stopwords')

# Step 6: Clean the text
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))  # remove special characters
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['clean_text'].astype(str).apply(clean_text)

# Step 7: Split the data into training and testing sets
X = df['clean_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Step 8: Convert text to numbers using CountVectorizer
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Step 9: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Step 10: Make predictions and evaluate
y_pred = model.predict(X_test_vect)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Display confusion matrix as heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
