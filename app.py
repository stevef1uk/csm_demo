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
    url = f"{ollama_url}/api/chat"
    
    # Clean up model name
    model_name = model_name.strip()
    if not model_name:
        error_msg = "Model name cannot be empty"
        logger.error(error_msg)
        return error_msg
    
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
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": message})
                conversation_history.append({"role": "assistant", "content": response_text})
                
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

def text_to_speech(text, speaker=0):
    """Convert text to speech using CSM"""
    logger.info(f"Converting text to speech with speaker {speaker}")
    try:
        # Log original text
        logger.info(f"Original text: '{text}'")
        
        # Clean and prepare the text
        text = text.strip()
        if not text:
            logger.warning("Empty text received for speech generation")
            return None
            
        # Log cleaned text
        logger.info(f"Cleaned text: '{text}'")
            
        # Prepare prompts with both speakers for better context
        logger.debug("Preparing speaker prompts...")
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
        logger.debug("Speaker prompts prepared successfully")
        
        # Generate audio with supported parameters
        logger.info("Generating audio...")
        logger.debug(f"Parameters: speaker={speaker}, max_length=30000ms")
        
        # Generate audio directly
        audio_tensor = generator.generate(
            text=text,
            speaker=speaker,
            context=[prompt_a, prompt_b],
            max_audio_length_ms=30_000  # Increased max length
        )
        
        # Make sure output directory exists
        os.makedirs("audio_outputs", exist_ok=True)
        
        # Use a timestamp for unique filename
        timestamp = int(time.time())
        output_path = f"audio_outputs/response_{timestamp}.wav"
        
        # Save to file using torchaudio directly
        torchaudio.save(
            output_path,
            audio_tensor.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        logger.info(f"Audio saved to {output_path}")
        
        # Calculate audio duration
        duration = audio_tensor.shape[0] / generator.sample_rate
        logger.info(f"Generated audio duration: {duration:.2f} seconds")
        
        # Verify the file exists and get info
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Audio file size: {file_size/1024:.2f} KB")
            
            # Also create a symlink or copy to a fixed path for easier replay
            fixed_path = "audio_outputs/latest_response.wav"
            try:
                if os.path.exists(fixed_path):
                    os.remove(fixed_path)
                os.link(output_path, fixed_path)
                logger.info(f"Created hard link to {fixed_path}")
            except Exception as e:
                logger.warning(f"Could not create link, copying file: {e}")
                import shutil
                shutil.copy2(output_path, fixed_path)
            
            # Return the absolute path to avoid any path resolution issues
            abs_path = os.path.abspath(output_path)
            logger.info(f"Returning absolute path: {abs_path}")
            return abs_path
        else:
            logger.error(f"Failed to save audio file to {output_path}")
            return None
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        logger.exception(e)
        return None

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
                
                # Convert speaker parameter to integer (0 for Man, 1 for Woman)
                speaker_id = 1 if speaker == "Woman" else 0
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
            
            # Convert speaker parameter to integer (0 for Man, 1 for Woman)
            speaker_id = 1 if speaker == "Woman" else 0
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
        
        # Convert speaker parameter to integer (0 for Man, 1 for Woman)
        speaker_id = 1 if speaker == "Woman" else 0
        
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
                choices=["Man", "Woman"],
                value="Man",
                label="Voice Type",
                info="Select the voice type for the AI response"
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
            
            if not transcription:
                return "", "No speech detected"
            
            logger.info(f"Transcribed: '{transcription}'")
            
            # We'll add this to conversation history in the LLM step
            # to avoid duplicates. Just show it to the user now.
            
            return transcription, "Processing with LLM..."
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            logger.exception(e)
            return "", f"Error: {str(e)}"
    
    # Second step: Process with LLM and generate audio
    def llm_process_step(transcription, model_name, ollama_url, speaker):
        if not transcription or transcription == "":
            return "No text to process", "", None
        
        try:
            # Get response from Ollama - this will add to conversation history
            logger.info(f"Processing with LLM: '{transcription}'")
            response = chat_with_ollama(transcription, model_name, ollama_url)
            
            if not response or response.startswith("Error"):
                return "Error getting AI response", response or "Error from LLM", None
            
            logger.info(f"LLM Response: '{response}'")
            
            # Convert speaker parameter to integer (0 for Man, 1 for Woman)
            speaker_id = 1 if speaker == "Woman" else 0
            
            # Generate audio response
            audio_path = text_to_speech(response, speaker_id)
            
            if not audio_path or not os.path.exists(audio_path):
                logger.error(f"Audio file not generated or doesn't exist: {audio_path}")
                return "Done (but audio failed)", response, None
            
            logger.info(f"Generated audio response: {audio_path}")
            
            return "Done!", response, audio_path
            
        except Exception as e:
            logger.error(f"Error processing with LLM: {str(e)}")
            logger.exception(e)
            return f"Error: {str(e)}", "", None
    
    # Process voice input in two steps
    # Step 1: Transcribe immediately
    voice_input.change(
        fn=transcribe_step,
        inputs=[voice_input],
        outputs=[transcribed_text, status]
    ).then(
        # Step 2: Process with LLM
        fn=llm_process_step,
        inputs=[transcribed_text, model_name, ollama_url, speaker],
        outputs=[status, text_output, voice_output]
    ).then(
        # Step 3: Update conversation display
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

if __name__ == "__main__":
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
        share=True,  # Enable public sharing
        debug=True,  # Enable debug mode for more information
        show_error=True,  # Show detailed error messages
        favicon_path=None,  # Disable favicon to avoid potential issues
        allowed_paths=None,  # Allow all paths for file uploads
        show_api=False,  # Don't show API documentation
        quiet=False  # Ensure we see the URL in the logs
    ) 
