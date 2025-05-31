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
from datasets import load_dataset
import torch
from TTS.api import TTS


# --- Load Environment Variables ---
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)

# --- Initialize Hugging Face Pipelines ---
logging.info("Loading Hugging Face pipelines...")
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1, token=HF_API_KEY)
llm_pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1,
    token=HF_API_KEY
)
# Initialize Coqui TTS model (once, globally)
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
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
        generated_text = llm_pipe(user_text, max_new_tokens=100, do_sample=True)[0]['generated_text']
        ai_text = generated_text.replace(user_text, "").strip()

        logging.info(f"AI Response: {ai_text}")

        # --- Text-to-Speech ---
        output_wav_path = "tts_output.wav"
        tts_model.tts_to_file(text=ai_text, file_path=output_wav_path)

        # Read audio and encode as base64 to return to frontend
        with open(output_wav_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
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
