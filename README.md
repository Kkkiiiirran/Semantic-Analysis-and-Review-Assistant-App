# Semantic Analysis and Support Assistant App

This Streamlit app predicts the **category of a user comment** (e.g., Support, Praise, Emotional, Toxic, Constructive Critisism, Spam etc.) and optionally generates a **positive reply** for the comment using **Google's Gemini API**. It supports **single comment input** as well as **batch prediction from CSV files**.  

---

## Features

1. **Pretrained sv` containing reviews and categories.  
   - Uses **DistilBERT embeddings*Model**  
   - Trained once on `main.c* and **Logistic Regression** for classification.  

2. **Single Comment Prediction**  
   - Input any comment in the text area.  
   - Predicts category and confidence.  
   - Generates a thoughtful positive reply using **Gemini API**.

3. **CSV Batch Prediction**  
   - Upload a CSV with a column `Review`.  
   - Predicts category for each review.  
   - Optionally generates positive replies for all reviews.  
   - Download results as CSV (`predictions_replies.csv`).

4. **Environment Variable Support**  
   - Gemini API key is loaded from a `.env` file for security.  

---

## Requirements

- Python 3.11V+
- Libraries:
  ```bash
  pip install streamlit pandas numpy scikit-learn torch transformers google-genai python-dotenv

## File Structure
```
project_folder/
│
├─ app.py              # Main Streamlit application
├─ model_setup.py      # Model training, loading, and prediction functions
├─ main.csv            # Training dataset (reviews + categories)
├─ .env                # API key for Gemini
├─ requirements.txt    # Required Python libraries
└─ README.md
```

## How to Use

### Single Comment

1. Enter the comment in the text area.  
2. The app predicts the category and displays confidence.  
3. A positive reply is generated using Gemini API.

### CSV Upload

1. Upload a CSV file with a column named `Review`.  
2. The app predicts the category for each review.  
3. Check the box to generate positive replies (optional).  
4. Download the resulting CSV.
