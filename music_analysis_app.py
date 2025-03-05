import streamlit as st
import librosa
import numpy as np
import openai
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os

# OpenAI API Key
openai.api_key = "your_api_key_here"

# Function to extract audio features
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = {
        "tempo": librosa.beat.tempo(y, sr=sr)[0],
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
        "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1).tolist()
    }
    return features

# Function to transcribe lyrics (if present)
def transcribe_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio.export(temp_wav.name, format="wav")
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_wav.name) as source:
        audio_data = recognizer.record(source)
        try:
            lyrics = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            lyrics = "Could not transcribe lyrics."
        except sr.RequestError:
            lyrics = "Speech recognition service unavailable."
    
    os.remove(temp_wav.name)
    return lyrics

# Function to get feedback from ChatGPT
def get_chatgpt_feedback(features, lyrics):
    prompt = f"""
    Analyze this song's musical characteristics:
    - Tempo: {features['tempo']} BPM
    - Spectral Centroid: {features['spectral_centroid']}
    - Zero Crossing Rate: {features['zero_crossing_rate']}
    - Timbre (MFCC values): {features['mfcc'][:5]}
    
    Lyrics (if available): {lyrics}
    
    Provide feedback on its mood, possible genre, and suggestions for improvement.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Streamlit App UI
st.title("AI Music Feedback App")
st.write("Upload your song to receive AI-generated feedback on its composition and lyrics.")

uploaded_file = st.file_uploader("Upload a song (MP3, WAV, etc.)", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name
    
    st.audio(temp_audio_path, format='audio/mp3')
    
    # Extract features
    st.write("Analyzing the song...")
    features = extract_audio_features(temp_audio_path)
    
    # Transcribe lyrics (if applicable)
    st.write("Transcribing lyrics...")
    lyrics = transcribe_audio(temp_audio_path)
    st.write("Extracted Lyrics:", lyrics)
    
    # Get ChatGPT Feedback
    st.write("Generating AI feedback...")
    feedback = get_chatgpt_feedback(features, lyrics)
    st.write("### AI Feedback:")
    st.write(feedback)
    
    # Cleanup temporary file
    os.remove(temp_audio_path)
