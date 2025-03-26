import os
import torch
import torchaudio
import gradio as gr
import requests
import json
import sounddevice as sd
import numpy as np
import soundfile as sf
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from generator import load_csm_1b, Segment
from huggingface_hub import hf_hub_download

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

# Initialize Whisper model
logger.info("Initializing Whisper model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
logger.info("Whisper model initialized successfully")

# Default prompts for voice generation
logger.info("Downloading prompt files...")
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)
logger.info("Prompt files downloaded successfully")

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

# Initialize the text-to-speech generator
logger.info("Initializing CSM generator...")
generator = load_csm_1b(device)
logger.info("CSM generator initialized successfully")

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    logger.debug(f"Loading prompt audio from {audio_path}")
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    logger.debug(f"Prompt audio loaded and resampled to {target_sample_rate}Hz")
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    logger.debug(f"Preparing prompt for speaker {speaker}")
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def chat_with_ollama(message, model_name, ollama_url):
    """Send message to Ollama and get response"""
    logger.info(f"Sending message to Ollama model: {model_name}")
    url = f"{ollama_url}/api/chat"  # Changed back to /api/chat endpoint
    
    # Clean up model name
    model_name = model_name.strip()
    if not model_name:
        error_msg = "Model name cannot be empty"
        logger.error(error_msg)
        return error_msg
    
    # Create a system prompt for conversational, concise responses
    system_prompt = "You are a friendly AI assistant. Keep your responses casual and conversational, using a maximum of two short sentences. Be concise and direct."
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "stream": False  # Disable streaming for simpler handling
    }
    
    try:
        # First check if Ollama is running
        try:
            logger.info(f"Checking Ollama server at {ollama_url}")
            version_response = requests.get(f"{ollama_url}/api/version")
            version_response.raise_for_status()
            logger.info("Ollama server is running")
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to Ollama server. Is it running?"
            logger.error(error_msg)
            return error_msg
        
        # Check if the model is available
        try:
            logger.info("Checking available models...")
            models_response = requests.get(f"{ollama_url}/api/tags")
            models_response.raise_for_status()
            available_models = [model["name"] for model in models_response.json().get("models", [])]
            logger.info(f"Available models: {available_models}")
            
            # Check if model exists
            if model_name not in available_models:
                error_msg = f"Model '{model_name}' not found. Available models: {', '.join(available_models)}. Please install it using 'ollama pull {model_name}'"
                logger.error(error_msg)
                return error_msg
                
            logger.info(f"Using model: {model_name}")
        except Exception as e:
            error_msg = f"Error checking available models: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
        # Send the request
        logger.info(f"Sending request to {url}")
        logger.debug(f"Request data: {json.dumps(data, indent=2)}")
        response = requests.post(url, json=data)
        response.raise_for_status()
        logger.info("Request sent successfully")
        
        # Parse the response
        try:
            result = response.json()
            logger.debug(f"Raw response: {json.dumps(result, indent=2)}")
            
            if "message" in result and "content" in result["message"]:
                # Clean up the response text
                response_text = result["message"]["content"]
                
                # Split into lines and clean each line
                lines = [line.strip() for line in response_text.splitlines() if line.strip()]
                
                # Join lines with spaces and clean up
                response_text = " ".join(lines)
                
                # Remove multiple spaces and normalize punctuation
                response_text = " ".join(response_text.split())
                
                # Remove emojis and special characters
                import re
                # Remove emojis
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emojis
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
                response_text = emoji_pattern.sub(r'', response_text)
                
                # Remove special characters but keep basic punctuation
                response_text = re.sub(r'[^\w\s.,!?]', '', response_text)
                
                # Clean up multiple punctuation marks
                response_text = re.sub(r'([.,!?])\1+', r'\1', response_text)
                
                # Ensure proper sentence ending
                if not response_text.endswith(('.', '!', '?')):
                    response_text += '.'
                
                logger.info(f"Successfully received and cleaned response: {response_text}")
                return response_text
            else:
                error_msg = "Response missing 'message' or 'content' field"
                logger.error(error_msg)
                logger.error(f"Full response: {result}")
                return error_msg
                
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Response content: {response.text}")
            return error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full error details: {e.__class__.__name__}: {str(e)}")
        return error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full error details: {e.__class__.__name__}: {str(e)}")
        return error_msg

def text_to_speech(text, speaker=0):
    """Convert text to speech using CSM"""
    logger.info(f"Converting text to speech with speaker {speaker}")
    try:
        # Clean and prepare the text
        text = text.strip()
        if not text:
            logger.warning("Empty text received for speech generation")
            return None
            
        # Prepare prompts with both speakers for better context
        prompt_a = prepare_prompt(
            SPEAKER_PROMPTS["conversational_a"]["text"],
            0,
            SPEAKER_PROMPTS["conversational_a"]["audio"],
            generator.sample_rate
        )
        prompt_b = prepare_prompt(
            SPEAKER_PROMPTS["conversational_b"]["text"],
            1,
            SPEAKER_PROMPTS["conversational_b"]["audio"],
            generator.sample_rate
        )
        
        # Generate audio with supported parameters
        logger.debug("Generating audio...")
        audio_tensor = generator.generate(
            text=text,
            speaker=speaker,  # Use the selected speaker
            context=[prompt_a, prompt_b],  # Use both prompts for better context
            max_audio_length_ms=15_000  # Increased max length for better quality
        )
        
        # Save to temporary file
        output_path = "temp_response.wav"
        torchaudio.save(
            output_path,
            audio_tensor.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        logger.info(f"Audio saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        raise

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

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        logger.info(f"Transcribing audio from {audio_path}")
        # Load and preprocess audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Normalize
        audio = audio / torch.max(torch.abs(audio))
        
        # Process audio with explicit English language setting
        inputs = processor(
            audio.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            language="en",
            task="transcribe"
        ).to(device)
        
        # Generate transcription
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=225,
                num_beams=5,
                language="en",
                task="transcribe"
            )
        
        transcription = processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Transcription result: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return f"Error transcribing audio: {str(e)}"

def chat_interface(message, model_name, ollama_url, speaker):
    """Main chat interface function"""
    logger.info("Processing chat interface request")
    try:
        # Get response from Ollama
        response = chat_with_ollama(message, model_name, ollama_url)
        
        # Convert response to speech with selected speaker
        audio_path = text_to_speech(response, speaker)
        
        logger.info("Chat interface request completed successfully")
        return response, audio_path
    except Exception as e:
        logger.error(f"Error in chat interface: {str(e)}")
        return f"Error: {str(e)}", None

def process_voice_input(audio_data, model_name, ollama_url, speaker):
    """Process voice input and return response"""
    logger.info("Processing voice input")
    try:
        if audio_data is None:
            logger.error("No audio data received")
            return "No audio data received", None, None
            
        # Convert audio data to the correct format
        if isinstance(audio_data, tuple):
            # If audio_data is a tuple (sample_rate, audio_array)
            sample_rate, audio_array = audio_data
        else:
            # If audio_data is just the array
            sample_rate = 16000
            audio_array = audio_data
            
        # Ensure audio is mono and float32
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        audio_array = audio_array.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Save the numpy array to a temporary file
        temp_file = "temp_input.wav"
        sf.write(temp_file, audio_array, sample_rate)
        logger.debug(f"Saved audio with shape {audio_array.shape} and sample rate {sample_rate}")
        
        # Convert speech to text
        text = transcribe_audio(temp_file)
        logger.debug(f"Transcribed text: {text}")
        
        # Get response from Ollama
        response = chat_with_ollama(text, model_name, ollama_url)
        logger.debug(f"Ollama response: {response}")
        
        # Convert response to speech with selected speaker
        response_audio = text_to_speech(response, speaker)
        logger.info("Voice input processing completed successfully")
        
        # Return transcribed text in message box, response in response box, and audio
        return text, response, response_audio
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        logger.error(f"Audio data type: {type(audio_data)}")
        if isinstance(audio_data, tuple):
            logger.error(f"Audio data shape: {audio_data[1].shape}")
        else:
            logger.error(f"Audio data shape: {audio_data.shape}")
        return f"Error: {str(e)}", None, None

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chat with AI")
    
    with gr.Row():
        with gr.Column():
            # Model selection
            model_name = gr.Textbox(
                label="Model Name",
                value="mistral:latest",
                info="Enter the name of the Ollama model to use (e.g., mistral:latest)"
            )
            # Ollama URL
            ollama_url = gr.Textbox(
                label="Ollama URL",
                value="http://192.168.1.53:11434",
                info="Enter the URL of your Ollama server"
            )
            # Speaker selection
            speaker = gr.Dropdown(
                choices=["Speaker A", "Speaker B"],
                value="Speaker A",
                label="Voice Type",
                info="Select the voice type for the AI response"
            )
            # Text input
            text_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                lines=3
            )
            # Voice input section with clear instructions
            gr.Markdown("### 🎤 Voice Input")
            gr.Markdown("Click the button below to start recording. Speak clearly and click again to stop.")
            voice_input = gr.Audio(
                type="numpy",
                label="Record Voice Message",
                show_label=True,
                show_download_button=False
            )
            # Submit button
            submit_btn = gr.Button("Send")
            
        with gr.Column():
            # Text output
            text_output = gr.Textbox(
                label="Response",
                lines=3,
                interactive=False
            )
            # Voice output
            voice_output = gr.Audio(
                label="Voice Response",
                type="numpy",
                interactive=False,
                autoplay=True  # Enable autoplay
            )

    # Wire up the interface
    submit_btn.click(
        fn=chat_interface,
        inputs=[text_input, model_name, ollama_url, speaker],
        outputs=[text_output, voice_output]
    )
    
    # Process voice input when audio is uploaded
    voice_input.change(
        fn=process_voice_input,
        inputs=[voice_input, model_name, ollama_url, speaker],
        outputs=[text_input, text_output, voice_output]  # Now includes message box as first output
    )

if __name__ == "__main__":
    # Create a log file for the URL
    with open("gradio_url.log", "w") as f:
        f.write("Gradio interface will be available at:\n")
        f.write("Local URL: http://0.0.0.0:7860\n")
        f.write("Public URL: Will be shown in the console when the server starts\n")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Enable public sharing
        debug=True,  # Enable debug mode for more information
        show_error=True,  # Show detailed error messages
        favicon_path=None,  # Disable favicon to avoid potential issues
        allowed_paths=None,  # Allow all paths for file uploads
        show_api=False,  # Don't show API documentation
        quiet=False  # Ensure we see the URL in the logs
    ) 
    
