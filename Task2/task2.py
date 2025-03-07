import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyDCdkAfSMz-FqB4YSXrXPPox2UNE8V39OE" #my google api key
genai.configure(api_key=GOOGLE_API_KEY)
torch.backends.cudnn.benchmark = True

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"  # Using 'stabilityai/stable-diffusion-2-1' for a different model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention()

    return pipe

pipe = load_model()
#pipe.save_pretrained("saved_model")  # Saves the model in 'saved_model' folder

# Function to generate Gemini AI text response or we can use GoogleAI model
def generate_text_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response.text else "🚫 Failed to generate response."
    except Exception as e:
        return f"🚫 Error processing text: {str(e)}"

def generate_stable_diffusion_image(prompt):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=10, guidance_scale=7.5).images[0]  # Adjusting steps for speed
    return image

# Streamlit UI
st.title("🤖 Multi-Modal Chatbot (Generate Text in 5sec or Image in 40sec)")

# User input
user_input = st.text_input("Enter text or 'generate image: [description]':")
submit = st.button("Send")

if submit and user_input:
    if user_input.lower().startswith("generate image:"):
        prompt = user_input.replace("generate image:", "").strip()
        with st.spinner("Generating Image... (Usually takes < 30s)"):
            image = generate_stable_diffusion_image(prompt)
        st.image(image, caption="🖼 Generated Image", use_container_width=True)
    else:
        with st.spinner("Generating Response..."):
            response = generate_text_response(user_input)
        st.write("💬 **Chatbot:**", response)
