# audio_utils.py
import os
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time

# Set up logger
logger = logging.getLogger(__name__)

# Initialize variables that will be set by initialize_whisper
processor = None
model = None
device = None

def initialize_whisper(device_to_use):
    """Initialize the Whisper model for speech-to-text"""
    global processor, model, device
    
    device = device_to_use
    logger.info("Initializing Whisper model...")
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
    
    logger.info("Whisper model initialized successfully")
    return processor, model

def transcribe_step(audio_data, session_id=None):
    """First step: Transcribe the audio"""
    if audio_data is None:
        logger.info("No audio data received")
        return "", "No audio recorded"
    
    try:
        # Start timing for speech recognition
        stt_start_time = time.time()
        logger.info("🎤 Starting speech-to-text conversion...")
        print("🎤 Starting speech-to-text conversion...")
        
        # Handle audio data
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            logger.info(f"Received audio for transcription: sr={sample_rate}, shape={audio_array.shape}")
        else:
            logger.error(f"Unexpected audio format: {type(audio_data)}")
            return "", "Error processing audio"
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
            audio_tensor = torch.FloatTensor(audio_array)
            if len(audio_tensor.shape) > 1:
                audio_tensor = torch.mean(audio_tensor, dim=1, keepdim=True).squeeze(1)
            
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=16000
            )
            audio_tensor = resampler(audio_tensor)
            audio_array = audio_tensor.numpy()
            sample_rate = 16000
        else:
            # Convert to mono if needed
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
        
        # Normalize audio
        if audio_array.size > 0 and np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Process with Whisper
        with torch.no_grad():
            inputs = processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(device)
            
            generated_ids = model.generate(
                inputs["input_features"],
                max_length=225,
                task="transcribe",
                language="en"
            )
            
            transcription = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
        
        # Log time taken for speech-to-text
        end_time = time.time()
        stt_elapsed = end_time - stt_start_time
        logger.info(f"⏱️ Speech-to-text conversion took {stt_elapsed:.2f} seconds")
        
        if not transcription:
            return "", "No speech detected"
        
        logger.info(f"Transcribed: '{transcription}'")
        
        # Add to conversation history (now with session_id)
        if session_id:
            from session_management import add_to_conversation
            add_to_conversation(session_id, "user", transcription)
        
        return transcription, f"Processing with LLM... (Transcription took {stt_elapsed:.2f}s)"
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        logger.exception(e)
        return "", f"Error: {str(e)}"

def replay_audio(audio_path):
    """Reload the audio file to force replay"""
    logger.info(f"Replaying audio from: {audio_path}")
    
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Audio file does not exist: {audio_path}")
        
        # Try using the fixed path as fallback
        fixed_path = "audio_outputs/latest_response.wav"
        if os.path.exists(fixed_path):
            logger.info(f"Using fallback path: {fixed_path}")
            return fixed_path
        return None
    
    # Return the absolute path
    abs_path = os.path.abspath(audio_path)
    logger.info(f"Returning absolute path for replay: {abs_path}")
    return abs_path

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    try:
        logger.info("Starting audio recording...")
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        logger.debug(f"Recording completed, shape: {recording.shape}")
        
        # Save to temporary file using soundfile
        output_path = "temp_input.wav"
        sf.write(output_path, recording, sample_rate)
        logger.info(f"Audio saved to {output_path}")
        
        # Return the file path as expected by Gradio
        return output_path
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        return None

def reset_for_recording():
    """Reset the interface for a new recording"""
    logger.info("Resetting for new recording")
    return None, "Ready for new recording. Click the microphone icon to start.", "", None

def debug_audio(audio_data):
    """Debug audio recording data"""
    print("🔎 DEBUG AUDIO CALLED")
    print(f"🔎 Audio data type: {type(audio_data)}")
    
    if audio_data is None:
        print("🔎 Audio is None")
        return "No audio recorded"
    
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sr, audio = audio_data
        print(f"🔎 Sample rate: {sr}, shape: {audio.shape}")
        if audio.size > 0:
            print(f"🔎 Audio range: {np.min(audio)} to {np.max(audio)}")
    
    return "Debug info printed to console"
