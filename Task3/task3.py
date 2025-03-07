import streamlit as st
import google.generativeai as genai
from langdetect import detect
from deep_translator import GoogleTranslator  #Replaced googletrans

#Directly Set Google API Key
GOOGLE_API_KEY = "AIzaSyCWo0oNkEc-wNZJHlzH6NzMoUSLfZIT-BI"

#Ensure API Key is Set
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Please check your API key.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
try:
    model = genai.GenerativeModel("gemini-pro")
    st.success(" Google API connected successfully!")
except Exception as e:
    st.error(f" Error initializing Google API: {str(e)}")

#Set API Key
genai.configure(api_key=GOOGLE_API_KEY)

#Initialize Google Gemini Model
try:
    model = genai.GenerativeModel("gemini-1.5-pro-latest") #The Google Gemini API (GenerativeModel) does not provide a method to save the model, as it is a cloud-based service.
    model.load_weights("SavedModelWt3.h5")
    model.save("saved_model#.h5")
    print("Model saved successfully as saved_model.h5")
    print("ModelWeights saved")
except Exception as e:
    st.error(f" Google API Error: {str(e)}")

#Streamlit UI
st.title("🌍 Multilingual AI Chatbot")
st.write("Chat with AI in your preferred language!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#UserInput
user_input = st.chat_input("Type your message...")

if user_input:
    detected_lang = detect(user_input)
    st.write(f"🔍 Detected Language: {detected_lang}")

    #Display user input in the output
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    #Translate to English
    translated_input = GoogleTranslator(source="auto", target="en").translate(user_input)
    st.write(f"🔹 Translated Input: {translated_input}")

    #Generate AI Response
    with st.spinner("Processing..."):
        try:
            response = model.generate_content(translated_input)
            bot_reply = response.candidates[0].content.parts[0].text  
            st.write(f" AI Response (English): {bot_reply}")
        except Exception as e:
            bot_reply = f" Error: {str(e)}"
            st.error(bot_reply)

    #Translate AI Response Back to Detected Language
    translated_reply = GoogleTranslator(source="en", target=detected_lang).translate(bot_reply)
    st.write(f"🌍 AI Response ({detected_lang}): {translated_reply}")
    st.session_state.messages.append({"role": "assistant", "content": translated_reply})

    with st.chat_message("assistant"):
        st.markdown(translated_reply)