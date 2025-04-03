import os
import time
import torch
import torchaudio
import logging
import numpy as np
from text_utils import sanitize_text_for_tts, log_timing

# Logger setup
logger = logging.getLogger(__name__)

# Speaker constants
SPEAKER_ID_WOMAN = 0  # UI selection "Woman" 
SPEAKER_ID_MAN = 1    # UI selection "Man"

# CSM Speaker mapping
CSM_SPEAKER_WOMAN = 0  # CSM model expects ID 0 for woman
CSM_SPEAKER_MAN = 1    # CSM model expects ID 1 for man

# Initialize with None - will be set from main app
generator = None

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
        "audio": None  # Will be set in initialize_tts
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
        "audio": None  # Will be set in initialize_tts
    }
}

def initialize_tts(generator_instance, prompt_a_path, prompt_b_path):
    """Initialize the TTS service with generator and prompt paths"""
    global generator, SPEAKER_PROMPTS
    generator = generator_instance
    
    # Set prompt audio paths
    SPEAKER_PROMPTS["conversational_a"]["audio"] = prompt_a_path
    SPEAKER_PROMPTS["conversational_b"]["audio"] = prompt_b_path
    
    logger.info("TTS service initialized with generator and prompt paths")

def load_prompt_audio(audio_path, target_sample_rate):
    """Load prompt audio from file"""
    logger.debug(f"Loading prompt audio from {audio_path}")
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    logger.debug(f"Prompt audio loaded and resampled to {target_sample_rate}Hz")
    return audio_tensor

def prepare_prompt(text, speaker, audio_path, sample_rate):
    """Prepare a prompt for TTS generation"""
    from generator import Segment
    logger.debug(f"Preparing prompt for speaker {speaker}")
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

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
        use_csm = True
        
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
        tts_elapsed = log_timing("Text-to-speech conversion", tts_start_time, logger)
        
        return f"Done! (TTS: {tts_elapsed:.2f}s)", audio_path
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        logger.exception(e)
        return f"Error generating audio: {str(e)}", None
