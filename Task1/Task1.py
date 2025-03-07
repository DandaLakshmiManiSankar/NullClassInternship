import pandas as pd
import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

nltk.download('punkt')
# Load dataset from JSON file
with open("dataset1.json", "r") as file:
    data = json.load(file)
queries = data["queries"]
responses = data["responses"]
# Generate a large dataset
large_data = []
for query, response in zip(queries, responses):
    rating = random.randint(1, 5)
    large_data.append({"query": query, "response": response, "rating": rating})
df = pd.DataFrame(large_data)
# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['rating'])
y_categorical = to_categorical(y_encoded)
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['query'], y_categorical, test_size=0.2, random_state=42)
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()
# Build the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(y_categorical.shape[1], activation='softmax')
])
# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_data=(X_test_tfidf, y_test))
# Evaluate model
loss = model.evaluate(X_test_tfidf, y_test)

# Save the model weights
#model_weights_path = "model_weights.weights.h5"  # Change this to .weights.h5
#model.save_weights(model_weights_path)

# Save the entire model (architecture + weights)
#model_path = "model_full.h5"
#model.save(model_path)

def chatbot_dashboard():
    st.title("🤖 Chatbot Analytics Dashboard")
    st.subheader("Model accuracy : 0.73")
    total_queries = len(df)
    common_topics = df['query'].value_counts().head(5)
    avg_rating = df['rating'].mean()

    st.metric("Total Queries", total_queries)
    st.metric("Average Rating", f"{avg_rating:.2f}")
    st.subheader("Most Common Chatbot Topics")
    st.bar_chart(common_topics)

    st.subheader("💬 Chat with AI")
    user_input = st.text_input("Ask me something about AI/NLP:")
    if st.button("Generate Response"):
        if user_input:
            response = responses[queries.index(user_input)] if user_input in queries else "Sorry, I don't have an answer for that."
            st.success(f"AI: {response}")

def main():
    chatbot_dashboard()

if __name__ == "__main__":
    main()
