import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- IMPORTANT ---
# You must have the *exact same* 'clean_text' function from your training script.
# You also need the NLTK data.

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
ps = PorterStemmer()

def clean_text(text):
    """The *exact same* cleaning function used during training."""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower()
    word_list = text.split()
    filtered_words = [word for word in word_list if word not in stop_words]
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

# --- Step 1: Load the Saved Model and Vectorizer ---
try:
    model = joblib.load('spam_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: 'spam_model.joblib' or 'tfidf_vectorizer.joblib' not found.")
    print("Please run the 'train.py' script first to create these files.")
    exit()

# --- Step 2: Create a Prediction Function ---
def is_spam(message) -> bool:
    # 1. Clean the new message
    cleaned_message = clean_text(message)
    
    # 2. Transform it using the *loaded* vectorizer
    vectorized_message = vectorizer.transform([cleaned_message])
    
    # 3. Predict using the *loaded* model
    prediction = model.predict(vectorized_message)
    
    return True if prediction[0] == 1 else False

if __name__ == '__main__':
  print("---------SPAM EMAIL DETECTOR---------")
  while True:
    message = input("Enter the message (Ctrl + C to Exit): ")
    if is_spam(message):
      print("Email is a Spam")
    else:
      print("Email isn't a Spam (Ham)")
  