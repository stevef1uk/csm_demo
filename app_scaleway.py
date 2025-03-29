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
import time
from openai import OpenAI
import sys

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

# Add conversation history
conversation_history = []

# Near the top of the file, add:
SCALEWAY_API_KEY = os.getenv("SCALEWAY_API_KEY", "")

# First, add these constants at the top of the file after the imports
# These will make our voice selection more explicit and consistent
SPEAKER_ID_WOMAN = 0  # UI selection "Woman" 
SPEAKER_ID_MAN = 1    # UI selection "Man"

# IMPORTANT: These were backwards! Swap them based on actual behavior
CSM_SPEAKER_WOMAN = 0  # CSM model expects ID 0 for woman
CSM_SPEAKER_MAN = 1    # CSM model expects ID 1 for man

# First, let's add a global variable to track the current service
CURRENT_SERVICE = "Scaleway"  # Default to Scaleway

def log_timing(operation_name, start_time):
    """Log the time taken for an operation"""
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    message = f"⏱️ {operation_name} took {elapsed_seconds:.2f} seconds"
    logger.info(message)
    print(message)
    return elapsed_seconds

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
    global conversation_history
    logger.info(f"Sending message to Ollama model: {model_name}")
    print(f"🔷 EXPLICITLY USING OLLAMA API with model: {model_name}")
    print(f"🔗 Connecting to Ollama URL: {ollama_url}")
    
    url = f"{ollama_url}/api/chat"
    
    # Start API request timer
    api_start_time = time.time()
    
    # Clean up model name
    model_name = model_name.strip()
    if not model_name:
        error_msg = "Model name cannot be empty"
        logger.error(error_msg)
        return error_msg
    
    # Log the URL we're trying to connect to
    logger.info(f"Full Ollama API URL: {url}")
    print(f"🔄 Making direct request to Ollama API at: {url}")
    
    # Create a system prompt for conversational, concise responses
    system_prompt = "You are a friendly AI assistant. Keep your responses casual and conversational, using a maximum of two short sentences. Be concise and direct."
    
    # Build messages array with conversation history
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # Add conversation history (last 5 exchanges)
    for msg in conversation_history[-10:]:  # Keep last 5 exchanges (10 messages)
        messages.append(msg)
    
    # Add current message
    messages.append({
        "role": "user",
        "content": message
    })
    
    data = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "timeout": 15  # Add timeout to request
    }
    
    try:
        # First check if Ollama is running with timeout
        try:
            logger.info(f"Checking Ollama server at {ollama_url}")
            version_response = requests.get(f"{ollama_url}/api/version", timeout=5)
            version_response.raise_for_status()
            logger.info("Ollama server is running")
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to Ollama server. Is it running?"
            logger.error(error_msg)
            return error_msg
        except requests.exceptions.Timeout:
            error_msg = "Connection to Ollama server timed out. Server might be busy."
            logger.error(error_msg)
            return error_msg
        
        # Check if the model is available with timeout
        try:
            logger.info("Checking available models...")
            models_response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            models_response.raise_for_status()
            available_models = [model["name"] for model in models_response.json().get("models", [])]
            logger.info(f"Available models: {available_models}")
            
            # Check if model exists
            if model_name not in available_models:
                # Try a fallback model if available
                if "mistral" in available_models:
                    logger.warning(f"Model '{model_name}' not found. Falling back to 'mistral'")
                    model_name = "mistral"
                elif len(available_models) > 0:
                    fallback_model = available_models[0]
                    logger.warning(f"Model '{model_name}' not found. Falling back to '{fallback_model}'")
                    model_name = fallback_model
                else:
                    error_msg = f"Model '{model_name}' not found. Available models: {', '.join(available_models)}. Please install it using 'ollama pull {model_name}'"
                    logger.error(error_msg)
                    return error_msg
                
            logger.info(f"Using model: {model_name}")
            # Update the model in the request data
            data["model"] = model_name
        except Exception as e:
            error_msg = f"Error checking available models: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
        # Send the request with timeout
        logger.info(f"Sending request to {url}")
        logger.debug(f"Request data: {json.dumps(data, indent=2)}")
        response = requests.post(url, json=data, timeout=30)  # 30 second timeout
        response.raise_for_status()
        logger.info("Request sent successfully")
        
        # Calculate and log API response time
        api_time = log_timing("Ollama API request", api_start_time)
        
        # Parse the response
        try:
            result = response.json()
            logger.debug(f"Raw response: {json.dumps(result, indent=2)}")
            
            if "message" in result and "content" in result["message"]:
                # Get the raw response text
                response_text = result["message"]["content"]
                logger.debug(f"Raw Ollama response: '{response_text}'")
                
                # Sanitize the response text
                response_text = sanitize_text_for_tts(response_text)
                logger.info(f"Sanitized Ollama response: '{response_text}'")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": message})
                conversation_history.append({"role": "assistant", "content": response_text})
                
                logger.info(f"Successfully received and cleaned response: {response_text}")
                print(f"📝 Ollama response received: {len(response_text)} characters in {api_time:.2f} seconds")
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
            
    except requests.exceptions.Timeout:
        error_msg = "Request to Ollama timed out. The server might be busy or the model is too large."
        logger.error(error_msg)
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

def text_to_speech(text, speaker=SPEAKER_ID_MAN):
    """Convert text to speech using CSM for short text and gTTS for longer text"""
    # Determine voice type based on UI selection
    voice_type = "Woman" if speaker == SPEAKER_ID_WOMAN else "Man"
    logger.info(f"🎤 Converting text to speech with voice type: {voice_type}")
    print(f"🎤 Converting text to speech with voice type: {voice_type}")
    
    try:
        # Log original text
        logger.info(f"Original text: '{text}'")
        
        # Clean and prepare the text
        text = text.strip()
        if not text:
            logger.warning("Empty text received for speech generation")
            return None
        
        # Check if gTTS is available
        gtts_available = False
        try:
            import gtts
            import pydub
            gtts_available = True
        except ImportError:
            logger.warning("gTTS or pydub not available, will use CSM for all responses")
            gtts_available = False
        
        # For CONSISTENCY: Use only one TTS method for the entire response
        # This prevents the half-woman/half-man issue on longer texts
        use_csm = True  # Default to CSM
        
        # ADJUSTED THRESHOLD: Use CSM for responses up to 300 characters
        # Set to exactly 300 as requested by user
        if gtts_available and len(text) > 300:
            use_csm = False
            logger.info(f"Using gTTS for longer text ({len(text)} chars)")
            print(f"💨 Using gTTS for voice: {voice_type}, text length: {len(text)} chars")
            
            # IMPROVED VOICE DISTINCTION: Use clearly different voices for gTTS
            if voice_type == "Woman":
                lang = "en"
                tld = "us"  # US English for Woman
                print(f"👩 Using gTTS with lang={lang}, tld={tld} for Woman voice")
            else:  # Man
                lang = "en"
                tld = "co.uk"  # UK English for Man 
                print(f"👨 Using gTTS with lang={lang}, tld={tld} for Man voice")
            
            try:
                from gtts import gTTS
                from pydub import AudioSegment
                
                # Create output directory
                os.makedirs("audio_outputs", exist_ok=True)
                
                # Generate unique filenames
                timestamp = int(time.time())
                mp3_path = f"audio_outputs/temp_{timestamp}.mp3"
                output_path = f"audio_outputs/response_{timestamp}.wav"
                
                logger.info(f"Generating gTTS speech with lang={lang}, tld={tld}")
                
                # Generate speech
                tts = gTTS(text=text, lang=lang, tld=tld)
                tts.save(mp3_path)
                
                # Convert to WAV
                sound = AudioSegment.from_mp3(mp3_path)
                sound.export(output_path, format="wav")
                
                # Clean up temp file
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
                
                # Create hardlink for latest response
                fixed_path = "audio_outputs/latest_response.wav"
                try:
                    if os.path.exists(fixed_path):
                        os.remove(fixed_path)
                    os.link(output_path, fixed_path)
                except Exception as e:
                    logger.warning(f"Could not create link: {e}")
                
                return output_path
            except Exception as e:
                logger.error(f"gTTS failed, falling back to CSM: {e}")
                print(f"⚠️ gTTS generation failed, falling back to CSM: {e}")
                use_csm = True  # Fall back to CSM
        
        # If we reach here, use CSM
        if use_csm:
            logger.info(f"Using CSM for speech generation ({voice_type} voice)")
            print(f"🎵 Using CSM for {voice_type} voice")
            
            # Prepare prompts
            prompt_a = prepare_prompt(
                SPEAKER_PROMPTS["conversational_a"]["text"],
                0,  # CSM prompt speaker 0
                SPEAKER_PROMPTS["conversational_a"]["audio"],
                generator.sample_rate
            )
            prompt_b = prepare_prompt(
                SPEAKER_PROMPTS["conversational_b"]["text"],
                1,  # CSM prompt speaker 1
                SPEAKER_PROMPTS["conversational_b"]["audio"],
                generator.sample_rate
            )
            
            # Use the correct CSM speaker ID based on our updated constants
            if voice_type == "Woman":
                csm_speaker = CSM_SPEAKER_WOMAN  # 0 for woman in CSM
                print(f"👩 Using CSM speaker ID {csm_speaker} for Woman voice")
            else:
                csm_speaker = CSM_SPEAKER_MAN  # 1 for man in CSM
                print(f"👨 Using CSM speaker ID {csm_speaker} for Man voice")
            
            # Generate audio
            audio_tensor = generator.generate(
                text=text,
                speaker=csm_speaker,
                context=[prompt_a, prompt_b],
                max_audio_length_ms=20_000,
                temperature=0.8
            )
            
            # Save to file
            os.makedirs("audio_outputs", exist_ok=True)
            timestamp = int(time.time())
            output_path = f"audio_outputs/response_{timestamp}.wav"
            
            torchaudio.save(
                output_path,
                audio_tensor.unsqueeze(0).cpu(),
                generator.sample_rate
            )
            
            # Create fixed path
            fixed_path = "audio_outputs/latest_response.wav"
            try:
                if os.path.exists(fixed_path):
                    os.remove(fixed_path)
                os.link(output_path, fixed_path)
            except Exception as e:
                logger.warning(f"Could not create link: {e}")
                import shutil
                shutil.copy2(output_path, fixed_path)
            
            return os.path.abspath(output_path)
        
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        logger.exception(e)
        return None

def generate_audio_for_response(status, response_text, speaker):
    """Generate audio for the response text separately"""
    if not response_text or response_text.startswith("Error"):
        return status, None
    
    try:
        # Start timing for text-to-speech
        tts_start_time = time.time()
        print("🔊 Starting text-to-speech conversion...")
        
        # Sanitize text for TTS
        sanitized_response = sanitize_text_for_tts(response_text)
        
        # Convert speaker parameter to integer (0 for Woman, 1 for Man)
        speaker_id = SPEAKER_ID_WOMAN if speaker == "Woman" else SPEAKER_ID_MAN
        
        # Generate audio response using sanitized text
        audio_path = text_to_speech(sanitized_response, speaker_id)
        
        # Log time taken for text-to-speech
        tts_elapsed = log_timing("Text-to-speech conversion", tts_start_time)
        
        return f"Done! (TTS: {tts_elapsed:.2f}s)", audio_path
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        logger.exception(e)
        return f"Error generating audio: {str(e)}", None

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

def transcribe_audio(audio_data, sample_rate=16000):
    """Transcribe audio using Whisper"""
    try:
        logger.info("Transcribing audio data")
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Process audio with explicit English language setting
        inputs = processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language="en",
            task="transcribe"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate transcription with proper attention mask
        with torch.no_grad():
            # Generate attention mask
            attention_mask = torch.ones_like(inputs["input_values"])
            
            # Generate transcription with attention mask
            outputs = model.generate(
                input_values=inputs["input_values"],
                attention_mask=attention_mask,
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
        logger.error(f"Audio data shape: {audio_data.shape}")
        logger.error(f"Audio data type: {audio_data.dtype}")
        logger.error(f"Audio data min: {np.min(audio_data)}, max: {np.max(audio_data)}")
        return f"Error transcribing audio: {str(e)}"

def chat_with_scaleway(message, model_name, api_key):
    """Send message to Scaleway LLM API and get response"""
    global conversation_history
    logger.info(f"EXPLICITLY USING SCALEWAY API with model: {model_name}")
    print(f"☁️ EXPLICITLY USING SCALEWAY API with model: {model_name}")
    
    if not api_key:
        error_msg = "Scaleway API key is required"
        logger.error(error_msg)
        return error_msg
    
    try:
        # Start API request timer
        api_start_time = time.time()
        
        # Verify we're using Scaleway API
        print(f"🔑 Using Scaleway API key: {api_key[:4]}...{api_key[-4:]}")
        
        # Initialize OpenAI client with Scaleway configuration
        client = OpenAI(
            base_url="https://api.scaleway.ai/e9873fc9-9fdb-4829-805a-cc706920d419/v1",
            api_key=api_key,
            timeout=60.0  # Increase timeout for reliability
        )
        
        # Create a system prompt for conversational, concise responses
        system_message = {
            "role": "system", 
            "content": "You are a friendly AI assistant. Keep your responses casual and conversational. Be concise and direct."
        }
        
        # Build messages array with conversation history
        messages = [system_message]
        
        # Add conversation history (last 5 exchanges)
        for msg in conversation_history[-10:]:
            messages.append(msg)
        
        # Add current message
        user_message = {"role": "user", "content": message}
        messages.append(user_message)
        
        logger.info(f"Sending request to Scaleway with {len(messages)} messages")
        
        # Make the API request
        try:
            print("🔄 Making direct request to Scaleway API...")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                stream=False
            )
            print("✅ Scaleway API responded successfully")
            
            # Calculate and log API response time
            api_time = log_timing("Scaleway API request", api_start_time)
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Sanitize the response text
            response_text = sanitize_text_for_tts(response_text)
            logger.info(f"Sanitized Scaleway response: '{response_text}'")
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": response_text})
            
            logger.info(f"📝 Scaleway response: '{response_text}'")
            print(f"📝 Scaleway response received: {len(response_text)} characters in {api_time:.2f} seconds")
            
            # Add verification that we used Scaleway
            print(f"☁️ Response confirmed from Scaleway API for model {model_name}")
            
            return response_text
        
        except Exception as api_error:
            logger.error(f"Error during Scaleway API request: {api_error}")
            print(f"❌ Error during Scaleway API request: {api_error}")
            raise api_error
            
    except Exception as e:
        error_msg = f"Error communicating with Scaleway: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return error_msg

def chat_interface(message, model_name, ollama_url, speaker):
    """Main chat interface function"""
    logger.info("Processing chat interface request")
    try:
        # Get response from Ollama
        response = chat_with_ollama(message, model_name, ollama_url)
        logger.debug(f"Ollama response: {response}")
        
        if not response or response.startswith("Error"):
            logger.error(f"Ollama response failed: {response}")
            return f"Error getting response: {response}", None
        
        # Convert response to speech with selected speaker
        response_audio = text_to_speech(response, speaker)
        logger.debug(f"Generated audio response")
        
        if not response_audio:
            logger.error("Failed to generate audio")
            return response, None
            
        logger.info("Chat interface request completed successfully")
        return response, response_audio
    except Exception as e:
        logger.error(f"Error in chat interface: {str(e)}")
        return f"Error: {str(e)}", None

def process_voice_input(audio_data, model_name, ollama_url, speaker):
    """Process voice input and return response"""
    logger.info("Processing voice input")
    try:
        # Handle empty input
        if audio_data is None:
            logger.debug("No audio data received")
            return "", "", None
            
        # Extract audio data and sample rate from Gradio input
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            logger.debug(f"Received audio: sample_rate={sample_rate}, shape={audio_array.shape}")
        else:
            logger.error(f"Unexpected audio_data format: {type(audio_data)}")
            return "Error: Unexpected audio format", "", None
        
        # Process audio data for transcription
        try:
            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
                audio_tensor = torch.FloatTensor(audio_array)
                if len(audio_tensor.shape) > 1:
                    # Convert stereo to mono by averaging channels
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
                
            logger.debug(f"Processed audio: shape={audio_array.shape}, sample_rate={sample_rate}")
            
            # Convert audio to text using Whisper
            with torch.no_grad():
                # Process audio with Whisper
                logger.debug("Processing with Whisper")
                inputs = processor(
                    audio_array,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to(device)
                
                # Generate transcription
                generated_ids = model.generate(
                    inputs["input_features"],
                    max_length=225,
                    task="transcribe",
                    language="en"
                )
                
                # Decode the transcription
                transcription = processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
                
                logger.info(f"Transcription: {transcription}")
                
                if not transcription:
                    logger.warning("Empty transcription result")
                    return "No speech detected", "", None
                    
                # Get response from Ollama
                logger.debug(f"Sending message to Ollama: {transcription}")
                response = chat_with_ollama(transcription, model_name, ollama_url)
                logger.info(f"Ollama response: {response}")
                
                if not response or response.startswith("Error"):
                    logger.error(f"Failed to get response from Ollama: {response}")
                    return transcription, response or "Error communicating with LLM", None
                
                # Convert speaker parameter to integer (0 for Woman, 1 for Man)
                speaker_id = 0 if speaker == "Woman" else 1
                logger.debug(f"Using speaker {speaker} (ID: {speaker_id})")
                
                # Generate audio response
                audio_response = text_to_speech(response, speaker_id)
                
                if audio_response is None:
                    logger.error("Failed to generate audio response")
                    return transcription, response, None
                
                logger.info("Successfully processed voice input")
                return transcription, response, audio_response
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            logger.exception(e)
            return f"Error processing audio: {str(e)}", "", None
            
    except Exception as e:
        logger.error(f"Voice input processing failed: {str(e)}")
        logger.exception(e)
        return f"Error: {str(e)}", "", None

def reset_conversation():
    """Reset the conversation"""
    global conversation_history
    conversation_history = []
    logger.info("Conversation history reset")
    return "", "Conversation reset. Ready for new messages.", "", None  # Reset transcribed text, status, response text, and audio

def transcribe_only(audio_data, model_name, ollama_url, speaker):
    """First stage: Just transcribe the audio"""
    if audio_data is None:
        logger.info("No audio data received")
        return "", "No audio recorded", "", None
        
    try:
        # Handle audio data
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            logger.info(f"Received audio: sample_rate={sample_rate}, shape={audio_array.shape}, min={np.min(audio_array)}, max={np.max(audio_array)}")
        else:
            logger.error(f"Unexpected audio_data format: {type(audio_data)}")
            return "", "Error: Unexpected audio format", "", None
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
            audio_tensor = torch.FloatTensor(audio_array)
            if len(audio_tensor.shape) > 1:
                # Convert stereo to mono by averaging channels
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
        transcription = ""
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
        
        logger.info(f"Transcription result: '{transcription}'")
            
        if not transcription:
            logger.warning("Empty transcription result")
            return "", "No speech detected", "", None
            
        # Update conversation history with user message immediately
        conversation_history.append({"role": "user", "content": transcription})
        
        # Immediately process with LLM
        logger.info("Processing transcription with LLM")
        llm_processing_status = "Processing your message..."
        
        try:
            response = chat_with_ollama(transcription, model_name, ollama_url)
            logger.info(f"LLM Response: '{response}'")
            
            if not response or response.startswith("Error"):
                logger.error(f"LLM response error: {response}")
                return transcription, "Error getting AI response", response or "Error communicating with LLM", None
            
            # Convert speaker parameter to integer (0 for Woman, 1 for Man)
            speaker_id = 0 if speaker == "Woman" else 1
            logger.debug(f"Using speaker {speaker_id} for voice response")
            
            # Generate audio response
            logger.info(f"Generating audio for: '{response}'")
            audio_response = text_to_speech(response, speaker_id)
            
            # Debug audio file
            if audio_response:
                if os.path.exists(audio_response):
                    file_size = os.path.getsize(audio_response)
                    duration = 0
                    try:
                        audio_info = sf.info(audio_response)
                        duration = audio_info.duration
                    except Exception as e:
                        logger.error(f"Error getting audio info: {e}")
                    
                    logger.info(f"Audio response file: {audio_response}, size: {file_size/1024:.2f}KB, duration: {duration:.2f}s")
                    
                    # Also create a simple test audio file
                    test_sine = np.sin(2 * np.pi * 440 * np.arange(0, 3, 1/16000))
                    sf.write("test_tone.wav", test_sine, 16000, subtype='PCM_16')
                    logger.info("Created test tone file: test_tone.wav")
                else:
                    logger.error(f"Audio response file not found: {audio_response}")
            else:
                logger.error("No audio response generated")
            
            logger.info("Successfully processed voice input and generated response")
            return transcription, "Done!", response, audio_response
            
        except Exception as e:
            logger.error(f"Error processing with LLM: {str(e)}")
            logger.exception(e)
            return transcription, f"Error in LLM processing: {str(e)}", "", None
            
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        logger.exception(e)
        return "", f"Error: {str(e)}", "", None

def process_transcription(transcription, status, model_name, ollama_url, speaker):
    """Second stage: Process the transcription with LLM and generate audio"""
    if not transcription or transcription == "":
        return status, "", None
        
    try:
        # Get response from Ollama
        logger.info(f"Processing transcribed text: {transcription}")
        response = chat_with_ollama(transcription, model_name, ollama_url)
        logger.info(f"LLM Response: {response}")
        
        if not response or response.startswith("Error"):
            return "Error getting AI response", response or "Error communicating with LLM", None
        
        # Convert speaker parameter to integer (0 for Woman, 1 for Man)
        speaker_id = 0 if speaker == "Woman" else 1
        
        # Generate audio response
        audio_response = text_to_speech(response, speaker_id)
        logger.info("Generated audio response successfully")
        
        if audio_response is None:
            return "Error generating audio response", response, None
        
        return "Done!", response, audio_response
    except Exception as e:
        logger.error(f"Error processing transcription: {str(e)}")
        logger.exception(e)
        return f"Error: {str(e)}", "", None

# Play button functionality - explicitly reload the file
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

def manual_record():
    """Record audio directly using sounddevice"""
    try:
        print("Starting manual recording...")
        logger.info("Starting manual recording...")
        
        # Record 5 seconds of audio
        sample_rate = 16000
        duration = 5  # seconds
        print(f"Recording {duration} seconds at {sample_rate}Hz...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # Wait until recording is finished
        
        print(f"Recording finished, shape: {recording.shape}")
        logger.info(f"Manual recording finished: shape={recording.shape}")
        
        # Normalize audio
        if np.max(np.abs(recording)) > 0:
            recording = recording / np.max(np.abs(recording))
            
        # Return as numpy tuple format that Gradio expects
        return (sample_rate, recording)
    except Exception as e:
        print(f"Error in manual recording: {e}")
        logger.error(f"Error in manual recording: {e}")
        logger.exception(e)
        return None
        
# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chat with AI")
    
    with gr.Row():
        with gr.Column(scale=1):
            # LLM Service selection - make it more prominent
            gr.Markdown("### Choose AI Service")
            llm_service = gr.Radio(
                choices=["Scaleway", "Ollama"],
                value="Scaleway",  # Default to Scaleway
                label="AI Service Provider",
                info="Select which service to use for AI responses",
                scale=1,
                min_width=300
            )
            
            # Model selection - more visible with a better heading
            gr.Markdown("### Select Model")
            model_name = gr.Dropdown(
                choices=[
                    # Scaleway models
                    "deepseek-r1-distill-llama-70b",
                    "meta-llama-3-70b-instruct",
                    "mixtral-8x7b-instruct-v0.1",
                    # Ollama models
                    "mistral:latest",
                    "llama3:8b",
                    "llama3:70b",
                    "gemma3:27b",
                    "phi3:14b",
                    "mixtral:8x7b",
                    "codellama:70b"
                ],
                value="deepseek-r1-distill-llama-70b",  # Default to Scaleway model
                label="AI Model",
                info="Select the model to use for responses",
                scale=1,
                min_width=300
            )
            
            # Voice type selection - change to Radio buttons for better visibility
            gr.Markdown("### Voice Settings")
            speaker = gr.Radio(  # Changed from Radio to make it more visible
                choices=["Woman", "Man"],  # Woman first for clarity
                value="Man",  # Keep default as Man
                label="Voice Type",
                info="Select the voice type for the AI response",
                scale=1,
                min_width=300
            )
            
            # Ollama URL - only visible when Ollama is selected
            ollama_url = gr.Textbox(
                label="Ollama Server URL",
                value="http://192.168.1.53:11434",  # Your custom IP
                info="Enter the URL of your Ollama server",
                visible=False,  # Initially hidden since Scaleway is default
                scale=1,
                min_width=300
            )
            
            # Voice input section with clear instructions
            gr.Markdown("### 🎤 Voice Input")
            gr.Markdown("Click the microphone icon to record, speak clearly, and click again to stop.")
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
    
    # Helper function to update conversation display
    def update_conversation_display():
        """Update the conversation history display"""
        if not conversation_history:
            return ""
        
        display_text = ""
        for msg in conversation_history:
            role = "You" if msg["role"] == "user" else "AI"
            display_text += f"{role}: {msg['content']}\n\n"
        
        return display_text
    
    # First step: Just transcribe and show immediately
    def transcribe_step(audio_data):
        if audio_data is None:
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
            stt_elapsed = log_timing("Speech-to-text conversion", stt_start_time)
            
            if not transcription:
                return "", "No speech detected"
            
            logger.info(f"Transcribed: '{transcription}'")
            
            return transcription, f"Processing with LLM... (Transcription took {stt_elapsed:.2f}s)"
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            logger.exception(e)
            return "", f"Error: {str(e)}"
    
    # First, add this utility function at the top with other utility functions
    def sanitize_text_for_tts(text):
        """Clean up text for TTS without removing standard punctuation"""
        import re
        if text is None:
            return ""
        
        # Remove emojis and other unicode special characters
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove control characters and non-printable characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Replace special quotes with standard quotes
        text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        
        # SPECIFICALLY REMOVE ASTERISKS AND EXCLAMATION POINTS 
        text = text.replace('*', '')
        text = text.replace('!', ' ')  # Replace exclamation marks with spaces
        
        # LESS AGGRESSIVE FILTERING: Allow common punctuation and symbols
        # Only remove truly problematic characters
        # Note: Removed asterisk (*) and exclamation (!) from the allowed characters
        text = re.sub(r'[^\w\s.,?()\'":;\-–—+&%$#@/\\|{}\[\]<>]', '', text)
        
        # Normalize whitespace (replace multiple spaces with a single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Only add ending punctuation if there's none already
        if text and not re.search(r'[.?]$', text):  # Removed ! from this check
            text += '.'
        
        return text

    # Step 1: Get LLM response only
    def get_llm_response(transcription, model_name, ollama_url, service=None):
        """Get LLM response only, without waiting for audio generation"""
        global CURRENT_SERVICE
        
        if not transcription or transcription == "":
            return "No text to process", ""
        
        try:
            # Start timing for LLM response
            llm_start_time = time.time()
            
            # Determine which service to use with fallback options
            # 1. Use explicitly provided service parameter if given
            # 2. Use global CURRENT_SERVICE if available
            # 3. Try to get from UI component
            # 4. Default to Scaleway as last resort
            current_service = service
            if current_service is None:
                current_service = CURRENT_SERVICE
            if current_service is None and hasattr(llm_service, 'value'):
                current_service = llm_service.value
            if current_service is None:
                current_service = "Scaleway"
            
            # Log which service and model we're using
            logger.info(f"🔷 Using service: {current_service}")
            logger.info(f"🔷 Model selected: {model_name}")
            print(f"🔷 Starting LLM request with {current_service} service and model: {model_name}")
            
            # CRITICAL FIX: FORCE LOG THE SERVICE AND URL
            if current_service == "Ollama":
                print(f"🤖 USING OLLAMA at URL: {ollama_url}")
            else:
                print(f"☁️ USING SCALEWAY with API key starting with: {SCALEWAY_API_KEY[:4]}...")
            
            # Get response from selected service - EXPLICITLY CHECK THE CURRENT SERVICE VALUE
            if current_service == "Scaleway":
                response = chat_with_scaleway(transcription, model_name, SCALEWAY_API_KEY)
            else:  # Ollama
                print(f"🔶 Attempting to connect to Ollama at: {ollama_url}")
                response = chat_with_ollama(transcription, model_name, ollama_url)
            
            # Log time taken for LLM response
            llm_elapsed = log_timing("LLM response", llm_start_time)
            
            # Return text response immediately
            return f"LLM response received ({llm_elapsed:.2f}s), generating audio...", response
        
        except Exception as e:
            logger.error(f"Error processing with LLM: {str(e)}")
            logger.exception(e)
            return f"Error: {str(e)}", ""

    # Add this function to directly update the global service variable
    def update_service_selection(service):
        """Update the global service selection variable"""
        global CURRENT_SERVICE
        old_service = CURRENT_SERVICE
        CURRENT_SERVICE = service
        print(f"🔄 Service changed from {old_service} to {CURRENT_SERVICE}")
        return service

    # Now update the toggle function inside the blocks context
    def toggle_service_options(service):
        """Handle service toggling between Scaleway and Ollama"""
        global CURRENT_SERVICE
        CURRENT_SERVICE = service  # Update global variable
        
        print(f"📢 Service toggled to: {service}")
        print(f"Global CURRENT_SERVICE is now: {CURRENT_SERVICE}")
        
        if service == "Scaleway":
            return gr.update(
                choices=[
                    "deepseek-r1-distill-llama-70b",
                    "meta-llama-3-70b-instruct", 
                    "mixtral-8x7b-instruct-v0.1"
                ],
                value="deepseek-r1-distill-llama-70b"
            ), gr.update(visible=False)
        else:  # Ollama
            return gr.update(
                choices=[
                    "mistral:latest",
                    "llama3:8b",
                    "llama3:70b",
                    "gemma3:27b",
                    "phi3:14b",
                    "mixtral:8x7b",
                    "codellama:70b"
                ],
                value="mistral:latest"
            ), gr.update(visible=True)

    # Process voice input in three steps
    voice_input.change(
        fn=transcribe_step,
        inputs=[voice_input],
        outputs=[transcribed_text, status]
    ).then(
        # First, explicitly update the service selection
        fn=update_service_selection,
        inputs=[llm_service],
        outputs=None
    ).then(
        # Step 2: Process with LLM and show text response immediately
        # Now explicitly passing the current service
        fn=lambda text, model, url, service: get_llm_response(text, model, url, service),
        inputs=[transcribed_text, model_name, ollama_url, llm_service],
        outputs=[status, text_output]
    ).then(
        # Step 3: Generate audio from the text response
        fn=generate_audio_for_response,
        inputs=[status, text_output, speaker],
        outputs=[status, voice_output]
    ).then(
        # Step 4: Update conversation display
        fn=update_conversation_display,
        inputs=None,
        outputs=[conversation_display]
    )
    
    # Record again button to clear the audio input
    record_again_btn.click(
        fn=lambda: None,  # Just return None to clear the audio input
        inputs=[],
        outputs=[voice_input]
    )
    
    # Reset conversation
    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[transcribed_text, status, text_output, voice_output]
    ).then(
        # Update conversation display after reset
        fn=lambda: "",
        inputs=None,
        outputs=[conversation_display]
    )
    
    # Add this debugging function to check the current state
    def debug_service_selection(service):
        """Print debug info about service selection"""
        print(f"👉 Service selected: {service}")
        return service

    # Make sure the event handler is properly connected
    llm_service.change(
        fn=toggle_service_options,
        inputs=[llm_service],
        outputs=[model_name, ollama_url]
    ).then(
        # Add this to validate selection was applied
        fn=update_service_selection,
        inputs=[llm_service],
        outputs=None
    )

# Launch code stays outside the blocks context
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("GRADIO_PORT", "7860"))
    
    # Create a log file for the URL
    with open("gradio_url.log", "w") as f:
        f.write("Gradio interface will be available at:\n")
        f.write(f"Local URL: http://0.0.0.0:{port}\n")
        f.write("Public URL: Will be shown in the console when the server starts\n")
    
    # Launch the interface with sharing enabled
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,  # Enable public sharing - CHANGED TO TRUE
        debug=True,
        show_error=True,
        favicon_path=None,
        allowed_paths=None,
        show_api=False,
        quiet=False
    ) 