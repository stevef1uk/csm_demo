"""
Complete Scaleway Voice Chat Application with Modal-style UI

This script creates a voice chat interface that matches app_scaleway_modal.py's UI:
1. Records audio from the microphone
2. Transcribes speech to text using Whisper
3. Sends text to Scaleway LLM API
4. Converts the response to speech using gTTS
"""

import os
import logging
import numpy as np
import time
import uuid
import json
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import requests
from gtts import gTTS

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add to top of your file to disable analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Global variables
api_key = os.getenv("SCALEWAY_API_KEY", "")
conversation_history = {}  # Dict to store conversations by session ID
SESSION_DIR = os.getenv("SESSION_DIR", "./user_sessions")
audio_dir = "audio_outputs"

# Create necessary directories
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

# Initialize Whisper model
def initialize_whisper():
    """Initialize the Whisper model for speech recognition."""
    logger.info("Initializing Whisper model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    logger.info(f"Whisper model initialized on {device}")
    return model, processor

# ==================== SESSION MANAGEMENT FUNCTIONS ====================

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
    global conversation_history
    
    # If session exists in memory, return it
    if session_id in conversation_history:
        return conversation_history[session_id]
    
    # Otherwise, try to load from disk
    try:
        session_path = get_session_path(session_id)
        if os.path.exists(session_path):
            with open(session_path, 'r') as f:
                conversation_history[session_id] = json.load(f)
            logger.info(f"Loaded session {session_id} from disk")
            return conversation_history[session_id]
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
    
    # If not found, initialize empty conversation
    conversation_history[session_id] = []
    return conversation_history[session_id]

def set_conversation_history(session_id, history):
    """Set conversation history for a specific session."""
    global conversation_history
    conversation_history[session_id] = history
    
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
        get_conversation_history(new_id)  # Load the conversation
        return new_id, f"Loaded existing session: {new_id}"
    else:
        # Initialize a new session
        set_conversation_history(new_id, [])
        return new_id, f"Created new session: {new_id}"

def save_user_session(session_id):
    """Save user session to disk."""
    try:
        history = get_conversation_history(session_id)
        session_path = get_session_path(session_id)
        
        with open(session_path, 'w') as f:
            json.dump(history, f)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
            
        logger.info(f"Session {session_id} saved to disk at {session_path}")
        return f"Session {session_id} saved successfully"
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")
        return f"Error saving session: {str(e)}"

def direct_load_session(name):
    """Direct session loader that just uses the exact name provided"""
    global conversation_history
    
    try:
        # Strip .json extension if provided
        if name.endswith('.json'):
            name = name[:-5]
            
        # Try to open the file directly 
        file_path = os.path.join(SESSION_DIR, f"{name}.json")
        
        if not os.path.exists(file_path):
            available = os.listdir(SESSION_DIR)
            logger.info(f"Direct file not found: {file_path}")
            logger.info(f"Available files: {available}")
            return f"Session '{name}' not found", ""
        
        # Read the file directly
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        
        # Update the conversation history for this session
        conversation_history[name] = session_data
        
        # Format for display
        display = ""
        for msg in session_data:
            role = "You" if msg["role"] == "user" else "AI"
            display += f"{role}: {msg['content']}\n\n"
        
        logger.info(f"Successfully loaded file: {file_path}")
        return f"Successfully loaded: {name}", display
    except Exception as e:
        logger.error(f"Direct file load error: {e}")
        return f"Error loading file: {str(e)}", ""

def clear_user_session(session_id):
    """Clear user session data."""
    try:
        set_conversation_history(session_id, [])
        return f"Session {session_id} cleared successfully"
    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {e}")
        return f"Error clearing session: {str(e)}"

# ==================== CORE FUNCTIONS ====================

def transcribe_step(audio_data, session_id):
    """Transcribe audio data using Whisper model."""
    logger.info(f"Starting speech-to-text conversion for session {session_id}")
    
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
            # Simple resampling
            audio_array = np.interp(
                np.linspace(0, len(audio_array), int(len(audio_array) * 16000 / sample_rate)),
                np.arange(len(audio_array)),
                audio_array
            )
            sample_rate = 16000
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Get global references to models
        global whisper_processor, whisper_model
        
        # Process with Whisper
        device = whisper_model.device
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

def get_llm_response(transcription, model_name, session_id):
    """Get response from Scaleway LLM API."""
    if not transcription:
        return "No text to process", "", None
    
    try:
        # Check API key
        global api_key
        if not api_key:
            return "Error: API key not found. Please check your setup.", "", None
        
        # Create request to Scaleway API
        url = "https://api.scaleway.ai/e9873fc9-9fdb-4829-805a-cc706920d419/v1/chat/completions"
        
        # Prepare conversation history
        conversation_history_for_session = get_conversation_history(session_id) if session_id else []
        
        messages = []
        
        # Add system message
        messages.append({
            "role": "system", 
            "content": "You are a helpful, respectful assistant."
        })
        
        # Add conversation history
        for msg in conversation_history_for_session:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        logger.info(f"Sending request to Scaleway API with {len(messages)} messages for session {session_id}")
        
        # Log first few characters of API key for verification (safely)
        if api_key:
            masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "******"
            logger.info(f"Using API key: {masked_key}")
        
        # Make the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        # Extract response text
        if response.status_code == 200:
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"].strip()
            
            # Add assistant response to history
            if session_id:
                add_to_conversation(session_id, "assistant", assistant_message)
            
            logger.info(f"LLM response for session {session_id}: '{assistant_message[:50]}...'")
            
            # Return text immediately to display
            return "Generating speech...", assistant_message, None
        else:
            logger.error(f"Error from API: {response.status_code} - {response.text}")
            return f"Error: API responded with {response.status_code}", "", None
    
    except Exception as e:
        logger.error(f"Error in LLM processing: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", "", None

def generate_speech(text):
    """Generate speech from text using gTTS."""
    try:
        if not text:
            return None
        
        logger.info(f"Generating speech for text: '{text[:50]}...'")
        
        # Create necessary directories
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate speech using gTTS
        timestamp = int(time.time() * 1000)
        mp3_path = f"{audio_dir}/response_{timestamp}.mp3"
        
        tts = gTTS(text=text, lang="en", tld="com")
        tts.save(mp3_path)
        
        logger.info(f"Generated audio at {mp3_path}")
        return mp3_path
            
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}", exc_info=True)
        return None

def update_conversation_display(session_id):
    """Update the conversation history display."""
    session_conversation = get_conversation_history(session_id)
    if not session_conversation:
        return ""
    
    display_text = ""
    for msg in session_conversation:
        role = "You" if msg["role"] == "user" else "AI"
        display_text += f"{role}: {msg['content']}\n\n"
    
    return display_text

def reset_for_recording():
    """Reset the interface for a new recording."""
    return None, "Ready for new recording", "", None

# ==================== GRADIO INTERFACE ====================

def create_gradio_interface():
    """Create Gradio interface for voice chat."""
    with gr.Blocks() as demo:
        gr.Markdown("# Scaleway Voice Chat")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Message about API key
                gr.Markdown("### API Access")
                gr.Markdown("API key is loaded from environment secret.")
                
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
                
                # Session Management
                gr.Markdown("### Session Management")
                
                session_id = gr.Textbox(
                    label="Session Name",
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
            fn=lambda session_id: ("", "Conversation reset", "", None),
            inputs=[session_id],
            outputs=[transcribed_text, status, text_output, voice_output]
        ).then(
            fn=clear_user_session,
            inputs=[session_id],
            outputs=[session_status]
        ).then(
            fn=update_conversation_display,
            inputs=[session_id],
            outputs=[conversation_display]
        )

    return demo

# Main execution
if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs(SESSION_DIR, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        # Log API key status (safely)
        if api_key:
            masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "******"
            logger.info(f"API key is set: {masked_key}")
        else:
            logger.warning("API key not set. The application may not function correctly.")
        
        # Initialize Whisper model
        whisper_model, whisper_processor = initialize_whisper()
        
        # Create and launch Gradio interface
        demo = create_gradio_interface()
        port = int(os.getenv("PORT", "7860"))
        
        logger.info(f"Starting Gradio server on port {port}")
        demo.launch(server_name="0.0.0.0", server_port=port)
        
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
