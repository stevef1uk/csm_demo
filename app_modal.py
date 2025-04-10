"""
Multimodal Voice Chat with Ollama LLM on Modal

This script deploys a voice chat interface on Modal that interacts with a remote Ollama server.
It provides speech-to-text, LLM processing, and text-to-speech capabilities in a single container.

Features:
- Speech-to-text using Whisper model
- Integration with remote Ollama LLM server
- Text-to-speech using CSM (Conditioned Sound Model)
- Conversation history management
- Modal proxy authentication for security
- GPU acceleration for ML tasks

Author: Steven Fisher (stevef@gmail.com)
"""

import modal
import torch
import torchaudio
import numpy as np
import requests
import json
import os
import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import gradio as gr
from gradio.routes import mount_gradio_app
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download, login
import soundfile as sf
from typing import Dict, List, Any, Optional, Union, Tuple
import sys
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("voice-chat-app-v1")
voice_chat_volume = modal.Volume.from_name("voice-chat-volume")  # Fixed volume name
server_url_secret = modal.Secret.from_name("llama_server_url")
proxy_token_id = modal.Secret.from_name("MODAL_PROXY_TOKEN_ID")
proxy_token_secret = modal.Secret.from_name("MODAL_PROXY_TOKEN_SECRET")
gradio_access_secret = modal.Secret.from_name("gradio_app_access_key")  # Added this secret
huggingface_token_secret = modal.Secret.from_name("hf-secret")  # Changed from "huggingface_token"

# Default GPU settings
DEFAULT_GPU_TYPE = "H100"
DEFAULT_GPU_COUNT = 1
DEFAULT_IDLE_TIMEOUT = 90  # 5 minutes

def get_gpu_config():
    """Get GPU configuration from environment variables or use defaults."""
    gpu_type = os.environ.get("GPU_TYPE", DEFAULT_GPU_TYPE)
    gpu_count_str = os.environ.get("GPU_COUNT", str(DEFAULT_GPU_COUNT))
    
    try:
        gpu_count = int(gpu_count_str)
    except ValueError:
        logger.warning(f"Invalid GPU count '{gpu_count_str}', using default: {DEFAULT_GPU_COUNT}")
        gpu_count = DEFAULT_GPU_COUNT
    
    valid_gpu_types = ["H100", "A100", "A100-40GB", "A100-80GB", "L40", "A10G"]
    
    if gpu_type not in valid_gpu_types:
        logger.warning(f"Unknown GPU type '{gpu_type}', falling back to {DEFAULT_GPU_TYPE}")
        gpu_type = DEFAULT_GPU_TYPE
    
    gpu_spec = gpu_type
    if gpu_count > 1:
        gpu_spec += f":{gpu_count}"
    
    logger.info(f"Using GPU configuration: {gpu_spec}")
    return gpu_spec

def get_idle_timeout():
    """Get idle timeout from environment variables or use default."""
    idle_timeout_str = os.environ.get("IDLE_TIMEOUT", str(DEFAULT_IDLE_TIMEOUT))
    
    try:
        idle_timeout = int(idle_timeout_str)
        if idle_timeout < 60:  # Minimum 1 minute
            logger.warning(f"Idle timeout too short '{idle_timeout_str}', using default: {DEFAULT_IDLE_TIMEOUT}")
            idle_timeout = DEFAULT_IDLE_TIMEOUT
    except ValueError:
        logger.warning(f"Invalid idle timeout '{idle_timeout_str}', using default: {DEFAULT_IDLE_TIMEOUT}")
        idle_timeout = DEFAULT_IDLE_TIMEOUT
    
    logger.info(f"Idle timeout: {idle_timeout} seconds ({idle_timeout/60:.1f} minutes)")
    return idle_timeout

# Global variables
conversation_history = []
models = {}

def check_silentcipher():
    import silentcipher
    print("Available attributes in silentcipher:", dir(silentcipher))
    try:
        # Try both potential import patterns
        try:
            from silentcipher import CSM
            print("CSM class is available")
        except ImportError:
            print("CSM class not found")
            
        try:
            from silentcipher import load_csm_1b
            print("load_csm_1b function is available")
        except ImportError:
            print("load_csm_1b function not found")
    except Exception as e:
        print(f"Error checking silentcipher: {str(e)}")

# Update the GENERATOR_MODULE to use the correct model loading approach
GENERATOR_MODULE = """
import os
import torch
from huggingface_hub import hf_hub_download, login
import silentcipher
import shutil
from pathlib import Path
import tempfile

# Define our own Segment class since silentcipher doesn't export it directly
class Segment:
    def __init__(self, text, speaker, audio):
        self.text = text
        self.speaker = speaker
        self.audio = audio

def load_csm_1b(device="cuda"):
    # First log into huggingface with token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"Logging into Hugging Face with provided token")
        login(token=hf_token)
    else:
        print("No Hugging Face token provided, will try anonymous download")
    
    # IMPORTANT: The actual model isn't in the expected paths we tried earlier
    # Instead, it's likely directly packaged in the silentcipher module
    # Let's try to load it directly using silentcipher's API
    try:
        print("Trying to load CSM model directly from silentcipher...")
        
        # Set environment variable to help silentcipher find models if needed
        models_dir = os.path.abspath("../Models")
        os.environ["SILENTCIPHER_MODEL_DIR"] = models_dir
        os.makedirs(os.path.join(models_dir, "44_1_khz", "73999_iteration"), exist_ok=True)
        
        # Here's how app.py is likely loading the model - directly via silentcipher
        from silentcipher import get_model
        
        # Try the simplest approach first
        model = get_model(model_type="44.1k", device=device)
        print("CSM model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model directly: {e}")
        try:
            # Try loading using a different approach - the model might be embedded in the package
            print("Trying alternate loading method...")
            
            # Check if we have the proper module available
            try:
                from silentcipher.model import csm
                print("Found silentcipher.model.csm module, trying to initialize...")
                model = csm.Model(device=device)
                print("CSM model initialized through alternate method!")
                return model
            except ImportError:
                print("silentcipher.model.csm module not available")
                
            # Try one more approach
            try:
                # Sometimes models will be in a package directory
                import pkg_resources
                package_dir = os.path.dirname(pkg_resources.resource_filename('silentcipher', '__init__.py'))
                print(f"silentcipher package directory: {package_dir}")
                
                # Look for model files in package directory
                model_files = []
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        if file.endswith('.pt') or file == 'hparams.yaml':
                            model_files.append(os.path.join(root, file))
                
                print(f"Found potential model files: {model_files}")
                
                # Try loading again after identifying files
                model = get_model(model_type="44.1k", device=device)
                print("CSM model loaded successfully after locating files!")
                return model
            except Exception as e2:
                print(f"Alternate loading failed: {e2}")
        except Exception as e3:
            print(f"All direct loading methods failed: {e3}")
            
        # Use fallback - create a gtts wrapper that matches the CSM interface
        print("Creating gTTS fallback wrapper")
        from gtts import gTTS
        import torchaudio
        import numpy as np
        
        class GTTSWrapper:
            def __init__(self):
                self.sample_rate = 24000
            
            def generate(self, text, speaker=0, context=None, max_audio_length_ms=30000):
                print(f"Generating speech with gTTS: {text}")
                # Create temp file
                temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                temp_mp3.close()
                
                # Generate MP3
                # Use different TLDs for slightly different voices
                tld = "us" if speaker == 0 else "co.uk"
                lang = "en"
                print(f"Using gTTS with lang={lang}, tld={tld} for speaker={speaker}")
                tts = gTTS(text=text, lang=lang, tld=tld)
                tts.save(temp_mp3.name)
                
                # Convert to tensor
                try:
                    import soundfile as sf
                    from pydub import AudioSegment
                    
                    # Convert MP3 to WAV
                    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_wav.close()
                    
                    sound = AudioSegment.from_mp3(temp_mp3.name)
                    sound.export(temp_wav.name, format="wav")
                    
                    # Load as tensor
                    audio_data, sr = torchaudio.load(temp_wav.name)
                    audio_tensor = audio_data[0]  # Get mono
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                        audio_tensor = resampler(audio_tensor)
                    
                    # Clean up
                    os.unlink(temp_mp3.name)
                    os.unlink(temp_wav.name)
                    
                    return audio_tensor
                except Exception as e:
                    print(f"Error converting audio: {e}")
                    # Fallback to zeros
                    return torch.zeros(self.sample_rate * 3)  # 3 seconds of silence
        
        return GTTSWrapper()
"""

# Then create the stub file
with open("generator_stub.py", "w") as f:
    f.write(GENERATOR_MODULE)

# Add this function to prepare the model directories
def prepare_model_directories():
    # Get the current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Get parent directory
    parent_dir = os.path.dirname(cwd)
    print(f"Parent directory: {parent_dir}")
    
    # Create the exact directory structure silentcipher is looking for
    expected_path = os.path.join(parent_dir, "Models", "44_1_khz", "73999_iteration")
    os.makedirs(expected_path, exist_ok=True)
    print(f"Created directory structure: {expected_path}")
    
    # Create a placeholder file to ensure the directory is recognized
    placeholder_path = os.path.join(expected_path, "placeholder.txt")
    with open(placeholder_path, "w") as f:
        f.write("Directory structure for silentcipher models")
    
    print(f"Created placeholder file at {placeholder_path}")
    
    # List contents for verification
    print(f"Contents of expected_path: {os.listdir(expected_path)}")

# Update the image definition to match your working environment
voice_chat_image = modal.Image.debian_slim().run_commands(
    # First install git and ffmpeg
    "apt-get update && apt-get install -y git ffmpeg",
    # Clone the CSM repository for proper model loading
    "git clone https://github.com/SesameAILabs/csm.git /opt/csm",
    # Make generator.py accessible
    "cp /opt/csm/generator.py /root/generator.py"
).pip_install(
    "transformers==4.49.0",
    "torch==2.4.0",
    "torchaudio==2.4.0",
    "numpy==1.26.0",
    "soundfile==0.13.1",
    "huggingface_hub==0.28.1",
    "gradio==5.23.1",
    "requests==2.31.0",
    "fastapi==0.115.12",
    "uvicorn==0.34.0",
    "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master",
    "moshi==0.2.2",
    "torchtune==0.4.0",
    "torchao==0.9.0",
    "tokenizers==0.21.0",
    "gtts==2.3.2",
    "pydub==0.25.1",
).env({
    "NO_TORCH_COMPILE": "1",
    "PYTHONPATH": "/opt/csm"  # Set directly without trying to append
})

# Define a volume to store models and audio files
voice_chat_volume = modal.Volume.from_name("voice-chat-volume", create_if_missing=True)

# Default prompts for voice generation
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
        "audio": None  # Will be filled during initialization
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
        "audio": None  # Will be filled during initialization
    }
}

# Create a persistent FastAPI app
web_app = FastAPI(
    title="Multimodal Voice Chat API",
    description="API for multimodal voice chat with Ollama LLM",
    version="1.0.0",
    root_path="",  # Ensure no root path is set to avoid routing issues
)

# Add CORS middleware to allow cross-origin requests
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a root route for health check that doesn't require authentication
@web_app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "ok", "message": "Voice Chat API is running"}

# Add a health check endpoint that doesn't require authentication
@web_app.get("/health")
async def health_check():
    """Health check endpoint that doesn't require authentication."""
    return {"status": "ok", "message": "Voice Chat API is healthy"}

class VoiceChatAPI:
    """Main API class that handles all voice chat functionality."""
    
    def __init__(self):
        self.whisper_model = None
        self.whisper_processor = None
        self.csm_generator = None
        self.speaker_prompts = None
        self.conversation_history = []
        self.TOKEN_ID = None
        self.TOKEN_SECRET = None
        
    async def startup(self):
        """Initialize on startup."""
        # Try to get token values from environment variables
        self.TOKEN_ID = (
            os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_ID") or
            os.environ.get("token_id") or
            os.environ.get("MODAL_PROXY_TOKEN_ID")
        )
        self.TOKEN_SECRET = (
            os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_SECRET") or
            os.environ.get("token_secret") or
            os.environ.get("MODAL_PROXY_TOKEN_SECRET")
        )
        
        logger.info(f"API initialized: Tokens available={bool(self.TOKEN_ID) and bool(self.TOKEN_SECRET)}")
        logger.info("Environment variables available:")
        for key in sorted(os.environ.keys()):
            if "token" in key.lower() or "secret" in key.lower():
                logger.info(f"  {key}: {bool(os.environ[key])}")
    
    def initialize_models(self):
        """Initialize all ML models in one place."""
        if not self.whisper_model:
            self._initialize_whisper()
        
        if not self.csm_generator:
            self._initialize_csm()
    
    def _initialize_whisper(self):
        """Initialize the Whisper model for speech recognition."""
        logger.info("Initializing Whisper model...")
        
        # Explicitly define cache paths on the mounted volume
        cache_dir = "/app/data/models/whisper"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if the model is already cached
        model_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
        if model_files:
            logger.info(f"Found cached Whisper model files: {len(model_files)} files")
        else:
            logger.info("No cached Whisper model found, will download")
        
        # Initialize Whisper model with explicit cache directory
        self.whisper_processor = WhisperProcessor.from_pretrained(
            "openai/whisper-base", 
            cache_dir=cache_dir,
            local_files_only=len(model_files) > 0  # Try to use cached files if they exist
        )
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base", 
            cache_dir=cache_dir,
            local_files_only=len(model_files) > 0
        )
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = self.whisper_model.to(device)
        
        logger.info(f"Whisper model initialized on {device}")
    
    def _initialize_csm(self):
        """Initialize the CSM TTS model."""
        try:
            logger.info("Starting CSM initialization...")
            
            # Create specific directories for CSM model caching
            csm_cache_dir = "/app/data/models/csm"
            os.makedirs(csm_cache_dir, exist_ok=True)
            
            # Set cache environment variables to help silentcipher find cached models
            os.environ["SILENTCIPHER_MODEL_DIR"] = csm_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = csm_cache_dir
            os.environ["HF_HOME"] = csm_cache_dir
            os.environ["NO_TORCH_COMPILE"] = "1"
            
            # Check if model files are already cached
            cached_model_files = []
            for root, dirs, files in os.walk(csm_cache_dir):
                for file in files:
                    if file.endswith('.ckpt') or file.endswith('.pt'):
                        cached_model_files.append(os.path.join(root, file))
            
            if cached_model_files:
                logger.info(f"Found {len(cached_model_files)} cached CSM model files")
                # List some files to verify
                for i, file in enumerate(cached_model_files[:5]):
                    logger.info(f"  Cached file {i}: {os.path.basename(file)}")
            else:
                logger.info("No cached CSM model files found, will download")
            
            # Rest of your CSM initialization code...
            
        except Exception as e:
            logger.error(f"Error initializing CSM: {str(e)}", exc_info=True)
            # Use the gTTS fallback as before
            # [fallback code]
    
    def prepare_prompt(self, text, speaker, audio_path, sample_rate):
        """Prepare a prompt for CSM generation."""
        # Use our own Segment class from generator module instead of silentcipher
        from generator import Segment
        
        logger.debug(f"Preparing prompt for speaker {speaker}")
        audio_tensor = self.load_prompt_audio(audio_path, sample_rate)
        return Segment(text=text, speaker=speaker, audio=audio_tensor)
    
    def load_prompt_audio(self, audio_path, target_sample_rate):
        """Load prompt audio from a file and resample if needed."""
        logger.debug(f"Loading prompt audio from {audio_path}")
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
        )
        logger.debug(f"Prompt audio loaded and resampled to {target_sample_rate}Hz")
        return audio_tensor
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper model."""
        logger.info("Processing audio for transcription")
        
        if audio_data is None:
            return "", "No audio recorded", "", None
            
        try:
            # Extract audio data and sample rate
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                sample_rate, audio_array = audio_data
            else:
                return "", "Error: Invalid audio format", "", None
            
            # Ensure audio is mono and at 16kHz
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
                
            # Initialize Whisper if needed
            if not self.whisper_model:
                self._initialize_whisper()
            
            # Process with Whisper
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.whisper_processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = self.whisper_model.generate(
                    inputs["input_features"],
                    max_length=225,
                    task="transcribe",
                    language="en"
                )
                
                transcription = self.whisper_processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
                
                if not transcription:
                    return "", "No speech detected", "", None
            
            logger.info(f"Transcription: '{transcription}'")
            
            # Clear the audio output before processing with LLM
            return transcription, "Processing with LLM...", "", None
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}", exc_info=True)
            return "", f"Error: {str(e)}", "", None
    
    def chat_with_llm(self, transcription, model_name, speaker=0):
        """Send transcription to LLM and get response."""
        if not transcription:
            return "No text to process", "", None
        
        try:
            # Check if tokens are available - try all possible environment variable names
            if not self.TOKEN_ID:
                self.TOKEN_ID = (
                    os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_ID") or
                    os.environ.get("token_id") or
                    os.environ.get("MODAL_PROXY_TOKEN_ID")
                )
            
            if not self.TOKEN_SECRET:
                self.TOKEN_SECRET = (
                    os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_SECRET") or
                    os.environ.get("token_secret") or
                    os.environ.get("MODAL_PROXY_TOKEN_SECRET")
                )

            if not self.TOKEN_ID or not self.TOKEN_SECRET:
                logger.error("Modal proxy tokens not found in any environment variable")
                return "Error: Modal proxy authentication tokens not available", "", None

            # Access the server URL - try all possible environment variable names
            server_url = (
                os.environ.get("LLAMA_SERVER_URL") or
                os.environ.get("MODAL_SECRET_LLAMA_SERVER_URL")
            )
            if not server_url:
                try:
                    server_url = server_url_secret.get()
                    logger.info("Retrieved server URL from secret directly")
                except Exception as secret_error:
                    logger.error(f"Failed to get server URL from secret: {str(secret_error)}")
                    return "Error: Server URL not found. Please check server_url_secret is correctly set.", "", None
            
            # Prepare request with the correct payload format
            headers = {
                "Content-Type": "application/json",
                "Modal-Key": self.TOKEN_ID,
                "Modal-Secret": self.TOKEN_SECRET
            }
            
            # Create a formatted context string with the entire conversation
            context = ""
            for msg in self.conversation_history:
                role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
                context += f"{role_prefix}{msg['content']}\n\n"
            
            # Create a prompt that includes context and the current question
            if context:
                # Add a system instruction to remember the conversation context
                full_prompt = (
                    "You are a helpful assistant having a conversation with a user. "
                    "The conversation so far:\n\n"
                    f"{context}\n"
                    f"User: {transcription}\n\n"
                    "Assistant: "
                )
                logger.info(f"Sending request with conversation context")
            else:
                full_prompt = f"User: {transcription}\n\nAssistant: "
                logger.info("Sending request with single prompt (no context)")
            
            # Use the simple format for all Ollama models
            payload = {
                "prompt": full_prompt,
                "temperature": 0.7,
                "model": model_name,
                "stream": False
            }
            
            logger.info(f"Sending request to LLM with model {model_name}")
            
            response = requests.post(
                server_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Check for 'response' field in Ollama's response format
            if 'response' in result:
                response_text = result['response'].strip()
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": transcription})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                logger.info(f"LLM response: '{response_text}'")
                
                # Return text immediately with a status message
                # This lets the user read the response while speech is generating
                return "Generating speech...", response_text, None
            else:
                return "Error: Invalid response format from LLM", "", None
        
        except Exception as e:
            logger.error(f"Error in LLM processing: {str(e)}", exc_info=True)
            return f"Error: {str(e)}", "", None
    
    def generate_speech(self, text, speaker=0):
        """Generate speech from text using CSM or fallback to gTTS."""
        try:
            if not text:
                return None
            
            logger.info(f"Generating speech for text: '{text}' with speaker {speaker}")
            
            # Use gTTS for longer messages (faster but lower quality)
            use_gtts = len(text) > 100  # Use gTTS for messages longer than 100 chars
            
            if use_gtts:
                logger.info(f"Using gTTS for faster response (message length: {len(text)})")
                # Fallback to Google TTS
                from gtts import gTTS
                from pydub import AudioSegment
                
                # Create necessary directories
                os.makedirs("/app/data/audio", exist_ok=True)
                
                # Generate speech using gTTS
                timestamp = int(time.time() * 1000)  # Use milliseconds for more uniqueness
                mp3_path = f"/app/data/audio/response_{timestamp}.mp3"
                wav_path = f"/app/data/audio/response_{timestamp}.wav"
                
                # Use different TLDs for slightly different voices
                tld = "us" if speaker == 0 else "co.uk"
                lang = "en"
                
                voice_type = "male" if speaker == 0 else "female"
                logger.info(f"Using gTTS with lang={lang}, tld={tld} for {voice_type} voice")
                
                tts = gTTS(text=text, lang=lang, tld=tld)
                tts.save(mp3_path)
                
                # Convert mp3 to wav for better compatibility with the app
                try:
                    sound = AudioSegment.from_mp3(mp3_path)
                    sound.export(wav_path, format="wav")
                    output_path = wav_path
                    os.remove(mp3_path)
                    
                    logger.info(f"Generated gTTS audio at {output_path}")
                    return output_path
                except Exception as conv_error:
                    logger.warning(f"Failed to convert MP3 to WAV: {str(conv_error)}")
                    return mp3_path  # Use MP3 directly if conversion fails
            else:
                # Use CSM for shorter messages where quality matters more
                # Initialize CSM model if needed
                if not self.csm_generator:
                    self._initialize_csm()
                
                # Prepare prompts
                prompt_a = self.prepare_prompt(
                    self.speaker_prompts["conversational_a"]["text"],
                    0,
                    self.speaker_prompts["conversational_a"]["audio"],
                    self.csm_generator.sample_rate
                )
                prompt_b = self.prepare_prompt(
                    self.speaker_prompts["conversational_b"]["text"],
                    1,
                    self.speaker_prompts["conversational_b"]["audio"],
                    self.csm_generator.sample_rate
                )
                
                # Generate audio with CSM - use shorter max_audio_length_ms for faster generation
                audio_tensor = self.csm_generator.generate(
                    text=text,
                    speaker=speaker,
                    context=[prompt_a, prompt_b],
                    max_audio_length_ms=20_000,  # Reduced from 30_000
                    temperature=0.8  # Slightly lower temperature for faster generation
                )
                
                # Save to file
                timestamp = int(time.time() * 1000)
                output_path = f"/app/data/audio/response_{timestamp}.wav"
                torchaudio.save(
                    output_path,
                    audio_tensor.unsqueeze(0).cpu(),
                    self.csm_generator.sample_rate
                )
                
                logger.info(f"Generated CSM audio at {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"All speech generation methods failed: {str(e)}")
            return None
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")
        return "", "Conversation reset", "", None
    
    def update_conversation_display(self):
        """Update the conversation history display."""
        if not self.conversation_history:
            return ""
        
        display_text = ""
        for msg in self.conversation_history:
            role = "You" if msg["role"] == "user" else "AI"
            display_text += f"{role}: {msg['content']}\n\n"
        
        return display_text
    
    # API request handlers
    async def process_voice(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice input and return transcription and LLM response with speech."""
        try:
            audio_data = request_data.get("audio_data")
            model_name = request_data.get("model_name", "llama3:8b")
            speaker = request_data.get("speaker", "Man")
            
            # Step 1: Transcribe audio
            transcription, status, _, _ = self.transcribe_audio(audio_data)
            
            if not transcription:
                return {
                    "success": False,
                    "error": status if "Error" in status else "Failed to transcribe audio",
                    "transcription": "",
                    "response_text": "",
                    "audio_path": None
                }
            
            # Step 2: Process with LLM
            speaker_idx = 1 if speaker == "Woman" else 0  # Convert to numeric value
            status, response_text, audio_path = self.chat_with_llm(transcription, model_name, speaker_idx)
            
            if "Error" in status:
                return {
                    "success": False,
                    "error": status,
                    "transcription": transcription,
                    "response_text": "",
                    "audio_path": None
                }
                
            # Return success response
            return {
                "success": True,
                "transcription": transcription,
                "response_text": response_text,
                "audio_path": audio_path,
                "conversation_history": self.conversation_history
            }
        
        except Exception as e:
            logger.error(f"Error processing voice request: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "transcription": "",
                "response_text": "",
                "audio_path": None
            }
    
    async def reset(self) -> Dict[str, Any]:
        """Reset conversation history."""
        try:
            self.reset_conversation()
            return {
                "success": True,
                "message": "Conversation reset successfully"
            }
        except Exception as e:
            logger.error(f"Error resetting conversation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Create singleton API instance
api_instance = VoiceChatAPI()

@web_app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    await api_instance.startup()
    
    # Add this: Initialize models at startup
    logger.info("Initializing models at startup...")
    api_instance.initialize_models()

# API endpoints in FastAPI
@web_app.post("/api/process_voice")
async def process_voice_endpoint(request_data: Dict[str, Any]):
    """API endpoint for processing voice input."""
    return await api_instance.process_voice(request_data)

@web_app.post("/api/reset_conversation")
async def reset_conversation_endpoint():
    """API endpoint for resetting conversation."""
    return await api_instance.reset()

# Add a redirect from root to UI
@web_app.get("/ui")
async def ui_redirect():
    """Redirect to the UI interface."""
    return RedirectResponse(url="/ui/")  # The trailing slash is important

# Gradio interface
def create_gradio_interface():
    """Create Gradio interface for voice chat."""
    with gr.Blocks() as demo:
        gr.Markdown("# Multimodal Voice Chat with Ollama LLM")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                model_name = gr.Dropdown(
                    choices=[
                        "gemma3:27b",
                        "llama3:8b",
                        "llama3:70b",
                        "phi3:14b",
                        "mistral:7b",
                        "mixtral:8x7b",
                        "codellama:70b"
                    ],
                    value="llama3:8b",
                    label="LLM Model",
                    info="Select the model to use for responses"
                )
                
                # Speaker selection
                speaker = gr.Dropdown(
                    choices=["Woman", "Man"],
                    value="Woman",
                    label="Voice Type",
                    info="Voice for audio responses"
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

        # Event handlers
        voice_input.change(
            fn=api_instance.transcribe_audio,
            inputs=[voice_input],
            outputs=[transcribed_text, status, text_output, voice_output]
        ).then(
            # Clear audio output before LLM processing to avoid confusion
            fn=lambda: (None),
            inputs=[],
            outputs=[voice_output]
        ).then(
            # Get LLM response first
            fn=lambda x, model, spk: api_instance.chat_with_llm(x, model, 0 if spk == "Woman" else 1),
            inputs=[transcribed_text, model_name, speaker],
            outputs=[status, text_output, voice_output]
        ).then(
            # Generate speech after LLM response is already displayed
            fn=lambda x, spk: api_instance.generate_speech(x, 0 if spk == "Woman" else 1),
            inputs=[text_output, speaker],
            outputs=[voice_output]
        ).then(
            # Update status when speech is ready
            fn=lambda: "Success!",
            inputs=[],
            outputs=[status]
        ).then(
            fn=api_instance.update_conversation_display,
            inputs=[],
            outputs=[conversation_display]
        )
        
        # Record again button
        record_again_btn.click(
            fn=lambda: None,
            inputs=[],
            outputs=[voice_input]
        )
        
        # Reset conversation
        reset_btn.click(
            fn=api_instance.reset_conversation,
            inputs=[],
            outputs=[transcribed_text, status, text_output, voice_output]
        ).then(
            fn=api_instance.update_conversation_display,
            inputs=[],
            outputs=[conversation_display]
        )

    return demo

# Mount Gradio app to FastAPI
try:
    mount_gradio_app(
        app=web_app,
        blocks=create_gradio_interface(),
        path="/ui",  # Mount Gradio under the /ui path instead of root
        app_kwargs={
            "title": "Voice Chat Interface",
            "favicon_path": None
        }
    )
    logger.info("Gradio interface mounted successfully at /ui")
except Exception as e:
    logger.error(f"Error mounting Gradio app: {str(e)}")

@app.function(
    image=voice_chat_image,
    gpu=get_gpu_config(),
    volumes={"/app/data": voice_chat_volume},
    secrets=[
        server_url_secret, 
        proxy_token_id, 
        proxy_token_secret, 
        gradio_access_secret,
        huggingface_token_secret  # Add the HF token secret
    ],
    scaledown_window=get_idle_timeout(),
    timeout=1800,
    max_containers=1,
)
@modal.asgi_app(requires_proxy_auth=True)
def serve():
    """Return the persistent ASGI app"""
    # Create necessary directories
    os.makedirs("/app/data/models", exist_ok=True)
    os.makedirs("/app/data/audio", exist_ok=True)
    os.makedirs("/app/data/csm_models", exist_ok=True)
    
    return web_app 