# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special import softmax
# import torch
# import matplotlib.pyplot as plt

# # Load RoBERTa model
# MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# # Sentiment labels
# labels = ['Negative', 'Neutral', 'Positive']
# emotions_map = {
#     'Negative': 'Sad 😢',
#     'Neutral': 'Neutral 😐',
#     'Positive': 'Happy 😊'
# }

# # Function to get sentiment scores
# def polarity_scores_roberta(example):
#     encoded_text = tokenizer(example, return_tensors='pt')
#     with torch.no_grad():
#         output = model(**encoded_text)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)

#     return {
#         'roberta_neg': round(float(scores[0]), 4),
#         'roberta_neu': round(float(scores[1]), 4),
#         'roberta_pos': round(float(scores[2]), 4)
#     }

# # Streamlit UI
# st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
# st.title("📊 Sentiment Analysis with RoBERTa")
# st.markdown("Enter any text to analyze its sentiment and view the emotion.")

# user_input = st.text_area("Enter your text here:", height=150)

# if st.button("Analyze Sentiment"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text to analyze.")
#     else:
#         with st.spinner("Analyzing..."):
#             scores = polarity_scores_roberta(user_input)

#             # Determine dominant sentiment
#             sentiment_scores = {
#                 'Negative': scores['roberta_neg'],
#                 'Neutral': scores['roberta_neu'],
#                 'Positive': scores['roberta_pos']
#             }
#             dominant = max(sentiment_scores, key=sentiment_scores.get)
#             emotion = emotions_map[dominant]

#             st.subheader("🔍 Sentiment Scores:")
#             st.write(sentiment_scores)

#             st.subheader("🧠 Emotion Detected:")
#             st.success(f"**{emotion}** (based on highest score: {dominant})")

#             # Optional: show bar chart
#             st.subheader("📈 Sentiment Distribution:")
#             st.bar_chart(sentiment_scores)

import torch

print(torch.__version__)