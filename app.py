# app.py
import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google.cloud import speech # type: ignore
from google.cloud import texttospeech # type: ignore
import google.generativeai as genai # type: ignore
from pydub import AudioSegment # type: ignore
import logging

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO) # Use INFO for general flow, DEBUG for more detail

if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in environment variables.")
    # Potentially raise an exception or exit if critical
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    # Potentially raise an exception or exit if critical

# --- Initialize Clients ---
try:
    # Google Cloud Clients (uses GOOGLE_APPLICATION_CREDENTIALS env var if set,
    # otherwise can infer credentials or use API key if configured correctly,
    # though service accounts are generally recommended for GCP)
    # Using API Key explicitly might require different client initialization depending on library version/setup.
    # For simplicity here, assuming default auth flow works or GOOGLE_APPLICATION_CREDENTIALS is set.
    # If using ONLY API keys, you might need specific client options.
    speech_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
    logging.info("Google Cloud clients initialized.")
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud clients: {e}")
    # Handle error appropriately

try:
    # Gemini Client
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Use the latest flash model
    logging.info("Gemini client initialized.")
except Exception as e:
    logging.error(f"Failed to initialize Gemini client: {e}")
    # Handle error appropriately


# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio_route():
    """Handles audio processing: STT -> LLM -> TTS"""
    if 'audio' not in request.files:
        logging.warning("No audio file found in request.")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    logging.info(f"Received audio file: {audio_file.filename}, mimetype: {audio_file.mimetype}")

    try:
        # --- 1. Speech-to-Text ---
        # Read audio data directly into memory
        audio_content = audio_file.read()

        # Optional: Use pydub to ensure compatibility or get format info
        # This adds robustness but also a dependency (and potentially ffmpeg)
        try:
            # Load audio using pydub to determine properties / potentially convert
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_content))
            sample_rate = audio_segment.frame_rate
            channels = audio_segment.channels
            logging.info(f"Audio properties (pydub): Sample Rate={sample_rate}, Channels={channels}, Format={audio_segment.format}")

            # Export to a format Google STT prefers (like LINEAR16 WAV) if necessary
            # For many common web formats (like webm/opus), Google might handle it directly
            # If direct handling fails, uncomment and adjust export format:
            # buffer = io.BytesIO()
            # audio_segment.export(buffer, format="wav") # Example: export as WAV
            # audio_content = buffer.getvalue()
            # encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16 # Match the export format
            # sample_rate = audio_segment.frame_rate # Use sample rate from pydub
            # logging.info("Converted audio to WAV for STT.")

            # Assuming direct handling for now (Google STT is quite flexible)
            # Determine encoding based on common browser mime types if possible
            encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS # Good default for webm/opus
            if 'audio/wav' in audio_file.mimetype:
                encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16 # Or WAV depending on specific wav format
                # For LINEAR16, sample_rate_hertz is often required by the API
                # Use pydub's detected rate
                sample_rate_hertz = sample_rate
            elif 'audio/ogg' in audio_file.mimetype and 'opus' in audio_file.mimetype:
                 encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
                 sample_rate_hertz = 48000 # Opus typically uses 48k, needed for OGG_OPUS
            else: # Default guess for webm/opus
                 encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
                 sample_rate_hertz = 16000 # Often sufficient for speech

            logging.info(f"Attempting STT with encoding: {encoding}, sample rate: {sample_rate_hertz}")


        except Exception as pydub_err:
            logging.warning(f"Pydub processing failed: {pydub_err}. Proceeding without explicit conversion.")
            # Fallback if pydub fails or isn't installed/configured
            encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS # Default guess
            sample_rate_hertz = 16000 # Default guess
            # Note: Google's API might auto-detect some formats even without explicit encoding

        # Configure STT request
        recognition_config = speech.RecognitionConfig(
            # encoding=encoding, # Let API try auto-detect first usually works well
            # sample_rate_hertz=sample_rate_hertz, # Required for some encodings like LINEAR16/FLAC
            language_code="en-US",  # Adjust language code as needed
            # model="latest_long", # Or choose a specific model, e.g., "telephony", "medical_dictation"
            enable_automatic_punctuation=True,
        )
        recognition_audio = speech.RecognitionAudio(content=audio_content)

        logging.info("Sending audio to Google Speech-to-Text API...")
        stt_response = speech_client.recognize(config=recognition_config, audio=recognition_audio)
        logging.info("Received response from Google Speech-to-Text API.")

        if not stt_response.results or not stt_response.results[0].alternatives:
            logging.warning("STT returned no transcription results.")
            user_text = "" # Or handle as an error condition
        else:
            user_text = stt_response.results[0].alternatives[0].transcript
            logging.info(f"Transcript: {user_text}")

        if not user_text.strip():
            logging.info("Empty transcript received. Skipping LLM and TTS.")
            return jsonify({
                "user_text": "[Silent or Unrecognized]",
                "ai_text": "",
                "audio_base64": None # No audio to send back
            })

        # --- 2. LLM Response Generation (Gemini) ---
        logging.info("Sending transcript to Gemini API...")
        # Basic prompt, you might want to add context or instructions
        prompt = f"User: {user_text}\nAI:"
        gemini_response = gemini_model.generate_content(prompt)
        logging.info("Received response from Gemini API.")

        # Handle potential safety blocks or empty responses
        try:
             ai_text = gemini_response.text
             logging.info(f"Gemini Response: {ai_text}")
        except ValueError:
             # Handle cases where the response might be blocked due to safety settings
             logging.warning("Gemini response blocked or empty. Checking parts...")
             ai_text = "[Response blocked or unavailable]"
             # You could inspect gemini_response.prompt_feedback or parts for details
             for part in gemini_response.parts:
                 # This is a simplified check, structure might vary
                 if hasattr(part, 'text') and part.text:
                     ai_text = part.text
                     logging.info(f"Using text from response part: {ai_text}")
                     break # Use the first available text part
        except Exception as gen_err:
             logging.error(f"Error extracting text from Gemini response: {gen_err}")
             ai_text = "[Error generating response]"


        # --- 3. Text-to-Speech ---
        logging.info("Sending AI text to Google Text-to-Speech API...")
        synthesis_input = texttospeech.SynthesisInput(text=ai_text)

        # Configure TTS voice (explore available voices in GCP docs)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            # name="en-US-Studio-M", # Example high-quality Studio voice
            # name="en-US-Wavenet-D", # Example WaveNet voice
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL # Or FEMALE/MALE
        )

        # Select the type of audio file you want
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3 # MP3 is widely compatible
        )

        tts_response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logging.info("Received audio response from Google Text-to-Speech API.")

        # Encode audio content to Base64 for easy JSON transfer
        audio_base64 = base64.b64encode(tts_response.audio_content).decode('utf-8')

        # --- 4. Send Response to Frontend ---
        return jsonify({
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_base64": audio_base64
        })

    except speech.exceptions.GoogleAPICallError as e:
        logging.error(f"Google Cloud API Error (Speech): {e}")
        return jsonify({"error": f"Speech API Error: {e.details()}"}), 500
    except texttospeech.exceptions.GoogleAPICallError as e:
         logging.error(f"Google Cloud API Error (TTS): {e}")
         return jsonify({"error": f"TTS API Error: {e.details()}"}), 500
    except Exception as e:
        logging.exception("An unexpected error occurred during processing.") # Logs traceback
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Use waitress or gunicorn for production deployments instead of app.run()
    app.run(debug=True, port=5001) # Use a different port if 5000 is common