"""
Simplified Scaleway Voice Chat Application for Modal with Session Management

This script deploys a voice chat interface on Modal that interacts with Scaleway's AI API.
It provides basic speech-to-text, LLM processing, and text-to-speech capabilities.
"""

import modal
import torch
import torchaudio
import numpy as np
import requests
import os
import time
import logging
import uuid
import json
from fastapi import FastAPI
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
from pydub import AudioSegment
import tempfile
from openai import OpenAI
from modal import enter, method

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("scaleway-voice-chat")
volume = modal.Volume.from_name("voice-chat-volume", create_if_missing=True)
scaleway_secret = modal.Secret.from_name("SCALEWAY_API_KEY")

# Global variables
api_key = None
session_conversations = {}  # Store conversations by session ID
SESSION_DIR = "/app/data/user_sessions"
SESSION_RETENTION_DAYS = 30
CLEANUP_INTERVAL_HOURS = 24

# Build a simple image with minimal dependencies
image = modal.Image.debian_slim().apt_install(
    "ffmpeg"  # Required for audio processing
).pip_install(
    "transformers==4.35.2",
    "torch==2.1.1",
    "torchaudio==2.1.1",
    "numpy==1.25.2",
    "gradio==5.23.1",
    "fastapi>=0.115.2",
    "uvicorn==0.24.0",
    "gtts==2.3.2",
    "pydub==0.25.1",
    "soundfile==0.12.1",
    "openai>=1.0.0",
)

# Create FastAPI app
web_app = FastAPI(title="Scaleway Voice Chat")

# ==================== SESSION MANAGEMENT FUNCTIONS ====================

def initialize_session_management():
    """Initialize session management system."""
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    # List existing session files for debugging
    try:
        session_files = os.listdir(SESSION_DIR)
        logger.info(f"Found {len(session_files)} existing session files: {session_files}")
    except Exception as e:
        logger.error(f"Error listing session files: {e}")
    
    logger.info(f"Session management initialized. Sessions stored in: {SESSION_DIR}")

def generate_fallback_session_id():
    """Generate a fallback session ID."""
    return str(uuid.uuid4())

def get_session_path(session_id):
    """Get the file path for a session."""
    # Sanitize session ID to be safe for file paths
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "._-")
    return os.path.join(SESSION_DIR, f"{safe_id}.json")

def get_conversation_history(session_id):
    """Get conversation history for a specific session."""
    global session_conversations
    
    # If session exists in memory, return it
    if session_id in session_conversations:
        return session_conversations[session_id]
    
    # Otherwise, try to load from disk
    try:
        session_path = get_session_path(session_id)
        if os.path.exists(session_path):
            with open(session_path, 'r') as f:
                session_conversations[session_id] = json.load(f)
            logger.info(f"Loaded session {session_id} from disk")
            return session_conversations[session_id]
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
    
    # If not found, initialize empty conversation
    session_conversations[session_id] = []
    return session_conversations[session_id]

def set_conversation_history(session_id, history):
    """Set conversation history for a specific session."""
    global session_conversations
    session_conversations[session_id] = history
    
    # Save to disk
    try:
        session_path = get_session_path(session_id)
        with open(session_path, 'w') as f:
            json.dump(history, f)
        logger.info(f"Saved session {session_id} to disk")
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")

def add_to_conversation(session_id, role, content):
    """Add a message to the conversation history."""
    history = get_conversation_history(session_id)
    history.append({"role": role, "content": content})
    set_conversation_history(session_id, history)
    return history

def apply_custom_session_id(custom_id):
    """Apply a custom session ID."""
    if not custom_id or custom_id.strip() == "":
        new_id = generate_fallback_session_id()
        return new_id, "Custom ID was empty, generated a new one"
    
    # Clean the ID for use
    new_id = "".join(c for c in custom_id if c.isalnum() or c in "._-")
    
    # Check if it already exists
    session_path = get_session_path(new_id)
    if os.path.exists(session_path):
        return new_id, f"Loaded existing session: {new_id}"
    else:
        # Initialize a new session
        set_conversation_history(new_id, [])
        return new_id, f"Created new session: {new_id}"

def save_user_session(session_id):
    """Save user session to disk."""
    try:
        # Get current session data
        history = get_conversation_history(session_id)
        
        # Ensure the session directory exists
        os.makedirs(SESSION_DIR, exist_ok=True)
        
        # Get the full path on the volume
        session_path = get_session_path(session_id)
        
        # Write to volume with explicit flush
        with open(session_path, 'w') as f:
            json.dump(history, f)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
            
        logger.info(f"Session {session_id} saved to volume at {session_path}")
        return f"Session {session_id} saved successfully to volume"
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")
        return f"Error saving session: {str(e)}"

def load_user_session(session_name):
    """Load user session directly from disk, ignoring any cached data"""
    try:
        # Always work with the raw name, no UUID conversion
        if session_name.endswith('.json'):
            session_name = session_name[:-5]
        
        # Try to find the file directly
        file_path = os.path.join(SESSION_DIR, f"{session_name}.json")
        
        if not os.path.exists(file_path):
            # List available sessions to help debugging
            available = [f.replace('.json', '') for f in os.listdir(SESSION_DIR)]
            logger.info(f"Available sessions: {available}")
            return f"Session '{session_name}' not found. Try one of: {', '.join(available)}", ""
        
        # Read directly from disk
        with open(file_path, 'r') as f:
            history = json.load(f)
        
        # Format for display
        display = ""
        for msg in history:
            role = "You" if msg["role"] == "user" else "AI"
            display += f"{role}: {msg['content']}\n\n"
        
        # Update the conversation after successful load
        global session_conversations
        session_conversations = {}  # Clear cache first
        session_conversations[session_name] = history
        
        return f"Session '{session_name}' loaded successfully", display
        
    except Exception as e:
        logger.error(f"Error loading session: {e}", exc_info=True)
        return f"Error: {str(e)}", ""

def clear_user_session(session_id):
    """Clear user session data."""
    try:
        set_conversation_history(session_id, [])
        return f"Session {session_id} cleared successfully"
    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {e}")
        return f"Error clearing session: {str(e)}"

def cleanup_old_sessions():
    """Clean up sessions older than SESSION_RETENTION_DAYS."""
    try:
        now = time.time()
        count = 0
        
        for filename in os.listdir(SESSION_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(SESSION_DIR, filename)
                file_age = now - os.path.getmtime(file_path)
                
                # Convert to days
                file_age_days = file_age / (60 * 60 * 24)
                
                if file_age_days > SESSION_RETENTION_DAYS:
                    os.remove(file_path)
                    count += 1
        
        logger.info(f"Cleaned up {count} old sessions")
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")

# ==================== WHISPER FUNCTIONS ====================

def initialize_whisper():
    """Initialize the Whisper model for speech recognition."""
    logger.info("Initializing Whisper model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    logger.info(f"Whisper model initialized on {device}")
    return processor, model

def transcribe_step(audio_data, session_id):
    """Transcribe audio data using Whisper model."""
    logger.info("Starting speech-to-text conversion")
    
    if audio_data is None:
        return "", "No audio recorded", "", None
        
    try:
        # Extract audio data and sample rate
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            logger.info(f"Received audio: sample_rate={sample_rate}Hz, shape={audio_array.shape}")
        else:
            return "", "Error: Invalid audio format", "", None
        
        # Ensure audio is mono
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            audio_tensor = torch.FloatTensor(audio_array)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            audio_array = audio_tensor.numpy()
            sample_rate = 16000
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Get global references to models
        global whisper_processor, whisper_model
        if 'whisper_processor' not in globals() or 'whisper_model' not in globals():
            whisper_processor, whisper_model = initialize_whisper()
        
        # Process with Whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = whisper_processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").to(device)
        
        start_time = time.time()
        with torch.no_grad():
            generated_ids = whisper_model.generate(
                inputs["input_features"],
                max_length=225,
                task="transcribe",
                language="en"
            )
            
            transcription = whisper_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
        
        end_time = time.time()
        logger.info(f"Transcription completed in {end_time - start_time:.2f}s: '{transcription}'")
        
        if not transcription:
            return "", "No speech detected", "", None
        
        # Add to conversation history
        if session_id:
            add_to_conversation(session_id, "user", transcription)
        
        return transcription, "Processing with LLM...", "", None
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}", exc_info=True)
        return "", f"Error: {str(e)}", "", None

# ==================== LLM FUNCTIONS ====================

def get_llm_response(transcription, model_name, session_id):
    """Get response from Scaleway LLM API."""
    if not transcription:
        return "No text to process", "", None
    
    try:
        # Check API key
        global api_key
        if not api_key:
            # Try one more time to get the key
            api_key_file = "/app/data/scaleway_key.txt"
            if os.path.exists(api_key_file):
                with open(api_key_file, "r") as f:
                    api_key = f.read().strip()
                logger.info("Retrieved API key from file")
            
            if not api_key:
                return "Error: API key not found. Please check your setup.", "", None
        
        # Create OpenAI client with Scaleway configuration
        client = OpenAI(
            base_url="https://api.scaleway.ai/e9873fc9-9fdb-4829-805a-cc706920d419/v1",
            api_key=api_key,
            timeout=60.0
        )
        
        # Prepare conversation history
        conversation_history = get_conversation_history(session_id) if session_id else []
        
        messages = []
        
        # Add system message
        messages.append({
            "role": "system", 
            "content": "You are a helpful, respectful assistant."
        })
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        logger.info(f"Sending request to Scaleway API with {len(messages)} messages")
        
        # Log first few characters of API key for verification (safely)
        if api_key:
            masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "******"
            logger.info(f"Using API key: {masked_key}")
        
        # Make the API request
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        
        # Extract response text
        assistant_message = response.choices[0].message.content.strip()
        
        # Add assistant response to history
        if session_id:
            add_to_conversation(session_id, "assistant", assistant_message)
        
        logger.info(f"LLM response: '{assistant_message[:50]}...'")
        
        # Return text immediately to display
        return "Generating speech...", assistant_message, None
    
    except Exception as e:
        logger.error(f"Error in LLM processing: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", "", None

# ==================== SPEECH FUNCTIONS ====================

def generate_speech(text):
    """Generate speech from text using gTTS."""
    try:
        if not text:
            return None
        
        logger.info(f"Generating speech for text: '{text[:50]}...'")
        
        # Create necessary directories
        os.makedirs("/app/data/audio", exist_ok=True)
        
        # Generate speech using gTTS
        timestamp = int(time.time() * 1000)
        mp3_path = f"/app/data/audio/response_{timestamp}.mp3"
        wav_path = f"/app/data/audio/response_{timestamp}.wav"
        
        tts = gTTS(text=text, lang="en", tld="com")
        tts.save(mp3_path)
        
        # Convert mp3 to wav for better compatibility
        try:
            sound = AudioSegment.from_mp3(mp3_path)
            sound.export(wav_path, format="wav")
            output_path = wav_path
            os.remove(mp3_path)
            
            logger.info(f"Generated audio at {output_path}")
            return output_path
        except Exception as conv_error:
            logger.warning(f"Failed to convert MP3 to WAV: {str(conv_error)}")
            return mp3_path  # Use MP3 directly if conversion fails
            
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}", exc_info=True)
        return None

# ==================== UTILITY FUNCTIONS ====================

def update_conversation_display(session_id):
    """Update the conversation history display."""
    conversation_history = get_conversation_history(session_id)
    if not conversation_history:
        return ""
    
    display_text = ""
    for msg in conversation_history:
        role = "You" if msg["role"] == "user" else "AI"
        display_text += f"{role}: {msg['content']}\n\n"
    
    return display_text

def reset_conversation(session_id):
    """Reset the conversation history."""
    set_conversation_history(session_id, [])
    logger.info(f"Conversation history reset for session {session_id}")
    return "", "Conversation reset", "", None

def reset_for_recording():
    """Reset the interface for a new recording."""
    return None, "Ready for new recording", "", None

def load_session_by_name(name):
    """Load a session by its name"""
    try:
        # Ensure we have a clean name without extension
        clean_name = name.replace('.json', '')
        
        # Create the file path
        file_path = os.path.join(SESSION_DIR, f"{clean_name}.json")
        
        if not os.path.exists(file_path):
            available = [f.replace('.json', '') for f in os.listdir(SESSION_DIR)]
            logger.warning(f"Session {clean_name} not found. Available: {available}")
            return f"Session not found. Available: {', '.join(available)}", ""
        
        # Load the session file
        with open(file_path, 'r') as f:
            history = json.load(f)
        
        # Update the session conversation
        global session_conversations
        session_conversations[clean_name] = history
        
        # Format for display
        display = ""
        for msg in history:
            role = "You" if msg["role"] == "user" else "AI"
            display += f"{role}: {msg['content']}\n\n"
        
        logger.info(f"Successfully loaded session {clean_name}")
        return f"Session {clean_name} loaded successfully", display
    
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        return f"Error: {str(e)}", ""

def direct_load_session(name):
    """Very simple direct session loader that just uses the exact name provided"""
    try:
        # Strip .json extension if provided
        if name.endswith('.json'):
            name = name[:-5]
            
        # Just try to open the file directly 
        file_path = os.path.join(SESSION_DIR, f"{name}.json")
        
        if not os.path.exists(file_path):
            available = os.listdir(SESSION_DIR)
            logger.info(f"Direct file not found: {file_path}")
            logger.info(f"Available files: {available}")
            return f"Session '{name}' not found. Available: {', '.join([f.replace('.json', '') for f in available])}", ""
        
        # Read the file directly
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Format for display
        display = ""
        for msg in data:
            role = "You" if msg["role"] == "user" else "AI"
            display += f"{role}: {msg['content']}\n\n"
        
        logger.info(f"Successfully loaded file: {file_path}")
        return f"Successfully loaded: {name}", display
    except Exception as e:
        logger.error(f"Direct file load error: {e}")
        return f"Error loading file: {str(e)}", ""

# ==================== GRADIO INTERFACE ====================

def create_gradio_interface():
    """Create Gradio interface for voice chat."""
    with gr.Blocks() as demo:
        gr.Markdown("# Scaleway Voice Chat")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Message about API key
                gr.Markdown("### API Access")
                gr.Markdown("API key is loaded from Modal secrets.")
                
                # Model selection
                gr.Markdown("### Model Selection")
                model_name = gr.Dropdown(
                    choices=[
                        "deepseek-r1-distill-llama-70b",
                        "meta-llama-3-70b-instruct",
                        "mixtral-8x7b-instruct-v0.1"
                    ],
                    value="deepseek-r1-distill-llama-70b",
                    label="AI Model",
                    info="Select the model to use for responses"
                )
                
                # Session Management - SIMPLIFIED SECTION
                gr.Markdown("### Session Management")
                
                session_id = gr.Textbox(
                    label="Session Name",  # Changed from "Session ID"
                    value=generate_fallback_session_id(),
                    info="Enter the session name to load or save (e.g. 'test1')"
                )
                
                custom_id_input = gr.Textbox(
                    label="Custom Session ID", 
                    placeholder="Enter a new name for this session",
                    info="Create a new name for this session"
                )
                
                with gr.Row():
                    apply_id_btn = gr.Button("Apply Custom ID")
                    save_session_btn = gr.Button("💾 Save Session")
                    load_session_btn = gr.Button("📂 Load Session")
                    clear_session_btn = gr.Button("🗑️ Clear Session")
                
                session_status = gr.Textbox(
                    label="Session Status",
                    value="New session created",
                    interactive=False
                )
                
                # Voice input section
                gr.Markdown("### 🎤 Voice Input")
                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Record Voice Message"
                )
                
                # Record again button
                record_again_btn = gr.Button("🎤 Record New Message")
                
                # Reset button
                reset_btn = gr.Button("Reset Conversation")
                
            with gr.Column(scale=2):
                # Transcribed text
                transcribed_text = gr.Textbox(
                    label="Your Message (Transcribed)",
                    lines=2,
                    interactive=False
                )
                
                # Processing status
                status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                
                # Conversation history display
                conversation_display = gr.Textbox(
                    label="Conversation History",
                    lines=8,
                    interactive=False
                )
                
                # Response output
                text_output = gr.Textbox(
                    label="AI Response",
                    lines=3,
                    interactive=False
                )
                
                # Voice output
                voice_output = gr.Audio(
                    label="Voice Response",
                    type="filepath",
                    interactive=False,
                    autoplay=True
                )
        
        # Voice input handler
        voice_input.change(
            fn=transcribe_step,
            inputs=[voice_input, session_id],
            outputs=[transcribed_text, status, text_output, voice_output]
        ).then(
            # Clear audio output before LLM processing
            fn=lambda: (None),
            inputs=[],
            outputs=[voice_output]
        ).then(
            # Get LLM response
            fn=get_llm_response,
            inputs=[transcribed_text, model_name, session_id],
            outputs=[status, text_output, voice_output]
        ).then(
            # Generate speech
            fn=generate_speech,
            inputs=[text_output],
            outputs=[voice_output]
        ).then(
            # Update status when done
            fn=lambda: "Success!",
            inputs=[],
            outputs=[status]
        ).then(
            fn=update_conversation_display,
            inputs=[session_id],
            outputs=[conversation_display]
        )
        
        # Session management handlers
        apply_id_btn.click(
            fn=apply_custom_session_id,
            inputs=[custom_id_input],
            outputs=[session_id, session_status]
        ).then(
            fn=update_conversation_display,
            inputs=[session_id],
            outputs=[conversation_display]
        )
        
        save_session_btn.click(
            fn=save_user_session,
            inputs=[session_id],
            outputs=[session_status]
        )
        
        # Use the direct_load_session function for the Load Session button
        load_session_btn.click(
            fn=direct_load_session,
            inputs=[session_id],
            outputs=[session_status, conversation_display]
        )
        
        clear_session_btn.click(
            fn=clear_user_session,
            inputs=[session_id],
            outputs=[session_status]
        ).then(
            fn=update_conversation_display,
            inputs=[session_id],
            outputs=[conversation_display]
        )
        
        # Record again button
        record_again_btn.click(
            fn=reset_for_recording,
            inputs=[],
            outputs=[voice_input, status, transcribed_text, text_output]
        )
        
        # Reset conversation
        reset_btn.click(
            fn=reset_conversation,
            inputs=[session_id],
            outputs=[transcribed_text, status, text_output, voice_output]
        ).then(
            fn=update_conversation_display,
            inputs=[session_id],
            outputs=[conversation_display]
        )

    return demo

# Mount Gradio app to FastAPI
gr.mount_gradio_app(
        app=web_app,
        blocks=create_gradio_interface(),
    path="/",
)

@app.function(
    image=image,
    volumes={"/app/data": volume},
    secrets=[scaleway_secret],
    timeout=1800,
    scaledown_window=90,
    max_containers=1,
    enable_memory_snapshot=True
)
@modal.asgi_app()
def app_entrypoint():
    """Return the ASGI app"""
    # Create necessary directories
    os.makedirs("/app/data", exist_ok=True)
    os.makedirs("/app/data/audio", exist_ok=True)
    
    # Initialize global variables
    global api_key
    
    # Initialize session management
    initialize_session_management()
    
    # Schedule session cleanup
    cleanup_old_sessions()
    
    # Try to get API key from various sources
    # 1. Check if a saved key exists in the volume
    api_key_file = "/app/data/scaleway_key.txt"
    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, "r") as f:
                api_key = f.read().strip()
            logger.info("API key loaded from volume")
        except Exception as e:
            logger.error(f"Error reading API key from file: {e}")
    
    # 2. Try environment variables as a backup
    if not api_key:
        env_vars = [
            "MODAL_SECRET_SCALEWAY_API_KEY",
            "SCALEWAY_API_KEY"
        ]
        for var in env_vars:
            if os.environ.get(var):
                api_key = os.environ.get(var)
                logger.info(f"API key loaded from environment variable: {var}")
                break
    
    # Log API key status
    if api_key:
        masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "******"
        logger.info(f"API key is set: {masked_key}")
    else:
        logger.warning("API key not found automatically, will need manual entry")
    
    # Initialize Whisper at startup
    global whisper_processor, whisper_model
    try:
        whisper_processor, whisper_model = initialize_whisper()
    except Exception as e:
        logger.error(f"Failed to initialize Whisper model: {e}")
    
    # In the app_entrypoint function, add this explicit path creation:
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    return web_app                  

@enter(snap=False)
def initialize_after_snapshot():
    """Reset session state after snapshot restore"""
    global session_conversations
    # Clear the in-memory cache to force fresh reads from disk
    session_conversations = {}
    logger.info("Session state reset after snapshot restore")

if __name__ == "__main__":
    print("Deploy this app with: modal deploy app_scaleway_modal.py")                                
