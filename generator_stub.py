
import os
import torch
from huggingface_hub import hf_hub_download
import silentcipher
import shutil
from pathlib import Path

class Segment:
    def __init__(self, text, speaker, audio):
        self.text = text
        self.speaker = speaker
        self.audio = audio

def load_csm_1b(device="cuda"):
    # Create the exact hardcoded directory structure that silentcipher expects
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Get parent directory
    parent_dir = os.path.dirname(cwd)
    print(f"Parent directory: {parent_dir}")
    
    # Create the exact directory structure
    expected_path = os.path.join(parent_dir, "Models", "44_1_khz", "73999_iteration")
    os.makedirs(expected_path, exist_ok=True)
    print(f"Created directory structure: {expected_path}")
    
    # Download directly to the expected directory
    try:
        # First let's try to see if silentcipher has a direct download function we can use
        print("Checking silentcipher attributes:", dir(silentcipher))
        
        # Load model with explicit debugging
        print("Attempting to load model...")
        model = silentcipher.get_model(
            model_type="44.1k",
            device=device
        )
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # List directories for debugging
        print(f"Contents of parent_dir: {os.listdir(parent_dir)}")
        models_dir = os.path.join(parent_dir, "Models")
        if os.path.exists(models_dir):
            print(f"Contents of Models dir: {os.listdir(models_dir)}")
            khz_dir = os.path.join(models_dir, "44_1_khz")
            if os.path.exists(khz_dir):
                print(f"Contents of 44_1_khz dir: {os.listdir(khz_dir)}")
        
        raise
