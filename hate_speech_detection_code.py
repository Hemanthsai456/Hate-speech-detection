import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

file_path = r"C:\Users\Hemanth Sai\OneDrive\Desktop\HateSpeechDataset.csv"
df = pd.read_csv(file_path, encoding='utf-8')

df = df.rename(columns={'Label': 'label', 'Content': 'text'})

df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

df.dropna(subset=['text'], inplace=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("Text Cleaning Done!")

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved!")

print("\nHate Speech Detector is ready! Type a sentence and press Enter.")

CONFIDENCE_THRESHOLD = 60

HATE_WORDS = {"bitch"}

def contains_hate_words(text):
    return any(word in text for word in HATE_WORDS)

while True:
    user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting... Thank you!")
        break

    if not user_input.strip():
        print("Please enter a valid sentence.")
        continue

    user_input_clean = clean_text(user_input)
    user_input_vec = vectorizer.transform([user_input_clean])
    prediction = model.predict(user_input_vec)
    confidence = model.predict_proba(user_input_vec)[0][prediction[0]] * 100

    if contains_hate_words(user_input_clean):
        print(f"⚠️ Hate Speech Detected! ")

    elif prediction[0] == 1:
        print(f"⚠️ Hate Speech Detected! ")
        
    else:
        print(f"✅ No Hate Speech. ")
