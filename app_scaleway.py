import os
import torch
import gradio as gr
import logging
import time
from huggingface_hub import hf_hub_download

# Import our custom modules
from audio_utils import (
    initialize_whisper, transcribe_step, replay_audio,
    record_audio, reset_for_recording, debug_audio
)
from llm_services import (
    update_service_selection, get_llm_response, toggle_service_options,
    CURRENT_SERVICE, set_api_key
)
from text_utils import (
    sanitize_text_for_tts, update_conversation_display, log_timing
)
from tts_service import (
    initialize_tts, generate_audio_for_response, SPEAKER_ID_WOMAN, SPEAKER_ID_MAN
)
from session_management import (
    initialize_session_management, save_user_session, load_user_session,
    clear_user_session, generate_fallback_session_id, apply_custom_session_id,
    set_conversation_history, get_conversation_history, add_to_conversation
)
from generator import load_csm_1b

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Select the best available device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Get Scaleway API key
SCALEWAY_API_KEY = os.getenv("SCALEWAY_API_KEY", "")
set_api_key(SCALEWAY_API_KEY)  # Set the key in the llm_services module

def initialize_application():
    """Initialize all application components"""
    logger.info("Initializing application components...")
    
    # Download prompt files
    logger.info("Downloading prompt files...")
    prompt_filepath_a = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav"
    )
    prompt_filepath_b = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_b.wav"
    )
    logger.info("Prompt files downloaded successfully")
    
    # Initialize Whisper model
    logger.info("Initializing Whisper model...")
    initialize_whisper(device)
    logger.info("Whisper model initialized successfully")
    
    # Initialize CSM generator
    logger.info("Initializing CSM generator...")
    generator = load_csm_1b(device)
    logger.info("CSM generator initialized successfully")
    
    # Initialize TTS service
    logger.info("Initializing TTS service...")
    initialize_tts(generator, prompt_filepath_a, prompt_filepath_b)
    logger.info("TTS service initialized successfully")
    
    # Create required directories
    os.makedirs("audio_outputs", exist_ok=True)
    
    return True

# Modified functions to include session ID
def transcribe_with_session(audio_data, session_id):
    """Wrapper for transcribe_step that includes session ID"""
    transcription, status = transcribe_step(audio_data)
    if transcription:
        # Add to the correct session's conversation history
        add_to_conversation(session_id, "user", transcription)
    return transcription, status

def get_llm_response_with_session(text, model_name, ollama_url, service, session_id):
    """Wrapper for get_llm_response that includes session ID"""
    status, response = get_llm_response(text, model_name, ollama_url, service, session_id)
    return status, response

def update_conversation_display_with_session(session_id):
    """Get the conversation display for a specific session"""
    conversation_history = get_conversation_history(session_id)
    display_text = ""
    for msg in conversation_history:
        role = "You" if msg["role"] == "user" else "AI"
        display_text += f"{role}: {msg['content']}\n\n"
    return display_text

def reset_conversation_with_session(session_id):
    """Reset the conversation for a specific session"""
    set_conversation_history(session_id, [])
    return "", "Conversation reset. Ready for new messages.", "", None, ""

def create_demo():
    """Create the Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# Chat with AI")
        
        with gr.Row():
            with gr.Column(scale=1):
                # LLM Service selection
                gr.Markdown("### Choose AI Service")
                llm_service = gr.Radio(
                    choices=["Scaleway", "Ollama"],
                    value="Scaleway",
                    label="AI Service Provider",
                    info="Select which service to use for AI responses",
                    scale=1,
                    min_width=300
                )
                
                # Model selection
                gr.Markdown("### Select Model")
                model_name = gr.Dropdown(
                    choices=[
                        # Scaleway models
                        "deepseek-r1-distill-llama-70b",
                        "meta-llama-3-70b-instruct",
                        "mixtral-8x7b-instruct-v0.1",
                    ],
                    value="deepseek-r1-distill-llama-70b",
                    label="AI Model",
                    info="Select the model to use for responses",
                    scale=1,
                    min_width=300
                )
                
                # Voice type selection
                gr.Markdown("### Voice Settings")
                speaker = gr.Radio(
                    choices=["Woman", "Man"],
                    value="Man",
                    label="Voice Type",
                    info="Select the voice type for the AI response",
                    scale=1,
                    min_width=300
                )
                
                # Ollama URL
                ollama_url = gr.Textbox(
                    label="Ollama Server URL",
                    value="http://192.168.1.53:11434",
                    info="Enter the URL of your Ollama server",
                    visible=False,
                    scale=1,
                    min_width=300
                )
                
                # Session Management
                gr.Markdown("### Session Management")
                
                session_id = gr.Textbox(
                    label="Session ID",
                    value=generate_fallback_session_id(),
                    info="Your unique session identifier"
                )
                
                custom_id_input = gr.Textbox(
                    label="Custom Session ID",
                    placeholder="Enter a memorable name",
                    info="Use a custom name for your session"
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
                gr.Markdown("Click the microphone icon to record, speak clearly, and click again to stop.")
                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Record Voice Message"
                )
                
                # Additional buttons
                record_again_btn = gr.Button("🎤 Record New Message")
                reset_btn = gr.Button("Reset Conversation")
                
            with gr.Column(scale=2):
                # Output components
                transcribed_text = gr.Textbox(
                    label="Your Message (Transcribed)",
                    lines=2,
                    interactive=False
                )
                
                status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                
                conversation_display = gr.Textbox(
                    label="Conversation History",
                    lines=8,
                    interactive=False
                )
                
                text_output = gr.Textbox(
                    label="AI Response",
                    lines=3,
                    interactive=False
                )
                
                voice_output = gr.Audio(
                    label="Voice Response",
                    type="filepath",
                    interactive=False,
                    autoplay=True
                )
        
        # Service selection handler
        llm_service.change(
            fn=toggle_service_options,
            inputs=[llm_service],
            outputs=[model_name, ollama_url]
        ).then(
            fn=update_service_selection,
            inputs=[llm_service],
            outputs=None
        )
        
        # Voice input handler - now passes session_id
        voice_input.change(
            fn=transcribe_with_session,
            inputs=[voice_input, session_id],
            outputs=[transcribed_text, status]
        ).then(
            fn=get_llm_response_with_session,
            inputs=[transcribed_text, model_name, ollama_url, llm_service, session_id],
            outputs=[status, text_output]
        ).then(
            fn=generate_audio_for_response,
            inputs=[status, text_output, speaker],
            outputs=[status, voice_output]
        ).then(
            fn=update_conversation_display_with_session,
            inputs=[session_id],
            outputs=[conversation_display]
        )
        
        # Session management handlers
        apply_id_btn.click(
            fn=apply_custom_session_id,
            inputs=[custom_id_input],
            outputs=[session_id, session_status]
        ).then(
            fn=update_conversation_display_with_session,
            inputs=[session_id],
            outputs=[conversation_display]
        )
        
        save_session_btn.click(
            fn=save_user_session,
            inputs=[session_id],
            outputs=[session_status]
        )
        
        load_session_btn.click(
            fn=load_user_session,
            inputs=[session_id],
            outputs=[session_status, conversation_display]
        )
        
        clear_session_btn.click(
            fn=clear_user_session,
            inputs=[session_id],
            outputs=[session_status]
        ).then(
            fn=update_conversation_display_with_session,
            inputs=[session_id],
            outputs=[conversation_display]
        )
        
        # Reset button handler
        reset_btn.click(
            fn=reset_conversation_with_session,
            inputs=[session_id],
            outputs=[transcribed_text, status, text_output, voice_output, conversation_display]
        )

        # Record new message button handler
        record_again_btn.click(
            fn=reset_for_recording,
            inputs=[],
            outputs=[voice_input, status, transcribed_text, text_output]
        )
    
    return demo

# Main execution
if __name__ == "__main__":
    # Initialize all components
    initialize_application()
    
    # Initialize session management
    initialize_session_management()
    
    # Create the Gradio interface
    demo = create_demo()
    
    # Get port from environment variable or use default
    port = int(os.getenv("GRADIO_PORT", "7860"))
    
    # Create a log file for the URL
    with open("gradio_url.log", "w") as f:
        f.write("Gradio interface will be available at:\n")
        f.write(f"Local URL: http://0.0.0.0:{port}\n")
        f.write("Public URL: Will be shown in the console when the server starts\n")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        debug=True,
        show_error=True
    ) 