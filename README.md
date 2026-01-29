# Fake News Detection

This project is a Machine Learning based Fake News Detection system that classifies news articles as **Fake** or **Real** using Natural Language Processing techniques. The trained ML model is integrated with a backend API and accessed through a simple web interface.

Fake news detection focuses on identifying misleading or false information by analyzing the writing style and word patterns present in news articles.  
Instead of relying on manual verification, this system uses Machine Learning to automatically learn differences between real and fake news from historical data.

The model captures important terms and phrases using TF-IDF, which helps highlight words that are more relevant in fake or real news. Based on these extracted features, the classifier predicts the authenticity of the news content provided by the user.

---

## Tech Stack
- Python  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- FastAPI (Backend)  
- Streamlit (Frontend)

---

## Project Structure
- `api.py` – Backend API for model prediction
- `app.py` – Streamlit frontend application
- `app.ipynb` – Model training and experimentation
- `Fake.csv` – Fake news dataset
- `True.csv` – Real news dataset
- `lr_model.jb` – Trained Logistic Regression model
- `vectorizer.jb` – TF-IDF vectorizer
- `requirements.txt` – Project dependencies

---

## Fake News Detection Approach
Fake news detection is performed by analyzing the textual content of news articles.  
The system focuses on identifying patterns in language usage that help distinguish fake news from real news.

- Text data is cleaned to remove noise such as punctuation and unnecessary symbols  
- TF-IDF is used to identify important words and phrases in the news content  
- Logistic Regression learns from labeled data to classify news articles  
- The model outputs both the predicted class (**Fake / Real**) and a confidence score  

---

## Backend & Frontend Integration
- The ML model is served using a FastAPI backend  
- User input from the frontend is sent to the backend for prediction  
- Backend processes the text and returns results  
- Predictions are displayed through a Streamlit interface  

---

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
