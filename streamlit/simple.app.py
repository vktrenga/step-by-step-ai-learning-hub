import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def correct_english(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are an English grammar / spelling correction assistant. Correct the following text while keeping its meaning."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def explain_more(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are an explanation assistant. Explain the topic with examples."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def generate_image(user_prompt):
    response = client.images.generate(
        model="gpt-image-1",
        prompt=user_prompt,
        size="1024x1024"
    )

    image_base64 = response.data[0].b64_json

    with open("generated_image.png", "wb") as f:
        import base64
        f.write(base64.b64decode(image_base64))

    return "generated_image.png"

def main():
    st.title("📝 English Grammar Correction / Explanation / Image Generation App")
    st.write("Enter a sentence for check English Grammar Correction / Explanation / Image Generation ")

    user_input = st.text_area("Your Text", height=100)

    # Create a single row with three buttons side by side
    col1, col2, col3 = st.columns(3)
    action = None

    with col1:
        if st.button("Correct Grammar"):
            action = "correct"

    with col2:
        if st.button("Explain More"):
            action = "explain"

    with col3:
        if st.button("Generate Image"):
            action = "image"

    # Show results at the bottom
    if action == "correct":
        if not user_input.strip():
            st.warning("Please enter some text to correct.")
        else:
            corrected_text = correct_english(user_input)
            st.subheader("✅ Corrected Text:")
            st.write(corrected_text)

    elif action == "explain":
        if not user_input.strip():
            st.warning("Please enter some text to explain.")
        else:
            explanation = explain_more(user_input)
            st.subheader("📘 Explanation:")
            st.write(explanation)

    elif action == "image":
        if not user_input.strip():
            st.warning("Please enter some text to generate an image.")
        else:
            image_url = generate_image(user_input)
            if image_url:
                st.subheader("🎨 Generated Image:")
                st.image(image_url, width=700)
            else:
                st.error("Image generation failed. Please check your API key or prompt.")
    
if __name__ == "__main__":
    main()
