import streamlit as st
import pandas as pd
from model_setup import train_and_save_model, load_model, predict_comment

import os
from google.generativeai import genai
from dotenv import load_dotenv

st.title("Semantic Analysis and Support Assistant")


load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

def generate_positive_reply(review_text):
    prompt = f"""
    Given the following customer review:
    '{review_text}'
    
    Please review the response. Then, try to understand the context.
    If its appreciation, support or constructive feedback, Please generate a thoughtful and positive reply that acknowledges the customer's satisfaction and expresses gratitude.
    Keep it short and personal.

    If its hate or spam or threat, repond with "we appreciate you took your time out to review. We request you to be mindful of how you convey your opinions."
    
    It shall only include a reply and nothing else. Don't mention things like:
    "Here is a feedback"
    "This is how you could reply"..
    etc.

    ONLY INCLUDE THE REPLY. NOTHING ELSE.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


MODEL_PATH = "clf.pkl"
MAIN_CSV = "main.csv"

if "clf" not in st.session_state:
    try:
        clf, tokenizer, model = load_model(MODEL_PATH)
        st.session_state.clf = clf
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.success("Model loaded successfully!")
    except:
        st.info("Training model on main.csv...")
        clf, tokenizer, model = train_and_save_model(MAIN_CSV, MODEL_PATH)
        st.session_state.clf = clf
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.success("Model trained and saved on main.csv!")


st.subheader("Predict Categories for CSV")
uploaded_file = st.file_uploader("Upload CSV with 'Review' column", type=["csv"], key="csv_predict")

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    if 'Review' not in df_input.columns:
        st.error("CSV must contain a 'Review' column")
    else:
        df_input['Category_Predicted'] = df_input['Review'].apply(
            lambda x: predict_comment(x, st.session_state.clf, st.session_state.tokenizer, st.session_state.model)[0]
        )
      
        if st.checkbox("Generate Positive Replies for all reviews"):
            df_input['Suggested_Reply'] = df_input['Review'].apply(generate_positive_reply)
        
        st.write(df_input)
        
      
        csv_exp = df_input.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", data=csv_exp, file_name="predictions_replies.csv", mime='text/csv')


st.subheader("Predict Single Comment and Generate Reply")
text_input = st.text_area("Enter comment here:")

if text_input:
    pred, conf = predict_comment(
        text_input,
        st.session_state.clf,
        st.session_state.tokenizer,
        st.session_state.model
    )
    st.write(f"**Predicted Category:** {pred}")
    if conf is not None:
        st.write(f"**Confidence:** {conf:.2f}")

  
    positive_reply = generate_positive_reply(text_input)
    st.write("**Suggested Positive Reply:**")
    st.write(positive_reply)
