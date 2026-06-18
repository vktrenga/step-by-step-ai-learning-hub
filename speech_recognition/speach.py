from email.mime import text
import os
import streamlit as st
import wave
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import os
from dotenv import load_dotenv
from openai import OpenAI
import pyttsx3

class SpeechRecognizer:
    def __init__(self):
        self.is_record_completed = False
        self.sample_rate = 44100  # CD quality
        self.input_text = ""
        self.output_text = ""
        self.recognizer = sr.Recognizer()
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)


    def recognize(self, audio_data):
        # Process the audio data and return the recognized text
        pass

    def record_audio(self, audio_file_path):
        st.title("🎙️ Audio Recorder with Streamlit & OpenAI Assistant : Explanation Bot")
        duration = st.slider("Recording duration (5 seconds)", 1, 5, 5)
        if st.button("Start Recording"):
            st.write("Recording...")
            audio_data = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished
            filename = "input.wav"
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
                st.success(f"Recording saved as {filename}")
                st.audio(filename)
                # self.read_audio_frames(filename)
                self.convert_audio_to_text_speech_recognition()

    def convert_audio_to_text_speech_recognition(self):
        st.write("convert_audio_to_text_speech_recognition started")
        with sr.AudioFile("input.wav") as source:
            audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio)
            st.write("Transcribed Text By SpeechRecognition:", text)
            self.input_text = text
            self.call_openai_api()

    def call_openai_api(self):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": "You are an explanation assistant. Explain the topic with examples."},
                    {"role": "user", "content": self.input_text}
                ]
            )
            st.markdown("### OpenAI API Response:")
            st.write(response.choices[0].message.content.strip())
            self.output_text = response.choices[0].message.content.strip()
            self.text_to_speach()
        except Exception as e:
            st.write(f"Error: {e}")

    def text_to_speach(self):
        engine = pyttsx3.init()
        engine.save_to_file(self.output_text, "output.mp3")
        engine.runAndWait()
        engine.stop()  # ✅ release resources
        st.markdown("### OpenAI API Response As Audio")
        st.audio("output.mp3")
       
  
objSpeech = SpeechRecognizer()
objSpeech.record_audio("output.wav")