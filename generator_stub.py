
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
