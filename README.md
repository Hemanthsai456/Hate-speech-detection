# Hate-speech-detection
# 🛑 Hate Speech Detection Using Machine Learning

A text classification project that identifies hate speech using machine learning and NLP techniques. This was built as part of my learning journey in AI & Data Science.

## 🚀 Overview

The internet can be a powerful tool — but it's also a place where harmful speech spreads easily.  
This project aims to detect hate speech from user-submitted text using a Logistic Regression model trained on real-world data.

Whether it's toxic comments, threats, or slurs — this model flags it.

## 🧠 What This Project Does

- Cleans and preprocesses raw text data
- Transforms text into numerical features using **TF-IDF**
- Trains a **Logistic Regression** classifier
- Predicts whether new text contains hate speech
- Flags known hate words directly
- Works in real time through console input

## 📁 Files in This Repo

| File | Purpose |
|------|---------|
| `hate_speech_detector.py` | Main Python script to train and run the model |
| `hate_speech_model.pkl` | Saved Logistic Regression model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF transformer |
| `requirements.txt` | Python libraries needed to run the project |
| `README.md` | You're reading it! 👋 |

## 📊 Sample Prediction

Enter a sentence: you are such a loser
⚠️ Hate Speech Detected!

Enter a sentence: have a nice day
✅ No Hate Speech.

