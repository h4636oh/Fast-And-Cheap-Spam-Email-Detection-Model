import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


local_csv_file = 'spam_dataset.csv'
print(f"Loading local data from {local_csv_file}...")
try:
    df = pd.read_csv(local_csv_file, encoding='latin-1')
except FileNotFoundError:
    print(f"Error: '{local_csv_file}' not found.")
    print("Please download the Kaggle CSV, rename it, and place it in the same folder.")
    exit()
except Exception as e:
    print(f"Failed to load data. Error: {e}")
    exit()

df = df[['CATEGORY', 'MESSAGE']]
df.columns = ['label', 'text']

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower()
    word_list = text.split()
    filtered_words = [word for word in word_list if word not in stop_words]
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

print("Cleaning and preprocessing text data...")
df['text_cleaned'] = df['text'].apply(clean_text)
print("Preprocessing complete.")

X = df['text_cleaned']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

print("Vectorizing text data using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Vectorizing complete.")

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
print("TF-IDF Vectorizer saved to 'tfidf_vectorizer.joblib'")

print("Training the Multinomial Naive Bayes model...")
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)
print("Model trained successfully!")

joblib.dump(model, 'spam_model.joblib')
print("Model saved to 'spam_model.joblib'")

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("------------------------\n")