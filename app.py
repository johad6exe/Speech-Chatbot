# app.py
import os
import io
import base64
import torch
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from transformers import pipeline
from pydub import AudioSegment

# --- Load Environment Variables ---
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)

# --- Initialize Hugging Face Pipelines ---
logging.info("Loading Hugging Face pipelines...")
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0 if torch.cuda.is_available() else -1, token=HF_API_KEY)
llm_pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",  # ✅ Replace here
    device=0 if torch.cuda.is_available() else -1,
    token=HF_API_KEY
)

tts_pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts", device=0 if torch.cuda.is_available() else -1, token=HF_API_KEY)
logging.info("Pipelines loaded successfully.")

# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_audio', methods=['POST'])
def process_audio_route():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    input_path = "temp_input.webm"
    wav_path = "temp_audio.wav"
    audio_file.save(input_path)

    try:
        # --- Convert to WAV (mono, 16kHz) ---
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        logging.info("Audio converted to WAV format for transcription.")

        # --- Speech-to-Text ---
        result = asr_pipe(wav_path)
        user_text = result['text'].strip()
        logging.info(f"Transcription: {user_text}")

        if not user_text:
            return jsonify({"user_text": "[Unrecognized speech]", "ai_text": "", "audio_base64": None})

        # --- LLM Response ---
        messages = [{"role": "user", "content": user_text}]
        llm_output = llm_pipe(messages, max_new_tokens=100)[0]['generated_text']
        ai_text = llm_output.replace(user_text, "").strip()
        logging.info(f"AI Response: {ai_text}")

        # --- Text-to-Speech ---
        tts_output = tts_pipe(ai_text)
        audio_bytes = tts_output["audio"]
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        logging.info("Generated TTS audio.")

        return jsonify({
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_base64": audio_base64
        })

    except Exception as e:
        logging.exception("Processing error")
        return jsonify({"error": str(e)}), 500

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
