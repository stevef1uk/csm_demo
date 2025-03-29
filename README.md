# Multimodal Voice Chat with AI Models

This application provides a voice chat interface powered by AI language models (Scaleway and Ollama). It features speech-to-text, LLM processing, and text-to-speech capabilities with support for multiple voice types.

## Features

- **Speech-to-Text**: Uses Whisper model to transcribe user voice input
- **Dual LLM Integration**: 
  - **Scaleway AI**: Connect to Scaleway's hosted LLM service
  - **Ollama**: Connect to a self-hosted Ollama server
- **Text-to-Speech**: Generates natural-sounding speech with:
  - **CSM (Conditioned Sound Model)**: High-quality voice for responses up to 300 characters
  - **gTTS (Google Text-to-Speech)**: Efficient fallback for longer responses
- **Voice Selection**: Choose between Woman and Man voices with consistent mapping
- **Conversation Memory**: Maintains context throughout the conversation
- **GPU Acceleration**: Optimized for ML tasks using CUDA if available
- **Responsive UI**: Clear interface with service selection and voice type options

## Prerequisites

### For Local Deployment (`app_scaleway.py`)

- Python 3.8+
- CUDA-compatible GPU (recommended for CSM performance)
- Scaleway API key (for Scaleway service)
- Ollama server (optional, for Ollama service)

### For Modal Deployment (`app_modal.py`)

- A [Modal](https://modal.com/) account
- An Ollama server endpoint for LLM access
- Modal CLI installed (`pip install modal`)
- Hugging Face account (for CSM model access)

## Setup Instructions for Local Deployment

### Step 1: Create a Python virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install torch torchaudio transformers gradio requests sounddevice numpy soundfile openai huggingface_hub
pip install silentcipher@git+https://github.com/SesameAILabs/silentcipher@master
pip install gtts pydub  # For gTTS fallback
```

### Step 3: Set environment variables

```bash
# Set your Scaleway API key
export SCALEWAY_API_KEY="your-scaleway-api-key"
```

### Step 4: Run the application

```bash
python app_scaleway.py
```

The application will be available at http://localhost:7860 by default.

## Setup Instructions for Modal Deployment

### Step 1: Install the Modal CLI

```bash
pip install modal
```

### Step 2: Configure Modal 

Login to Modal:

```bash
modal token new
```

### Step 3: Create Required Secrets

Create the following secrets in your Modal account:

```bash
# URL for your Ollama server
modal secret create llama_server_url "https://your-ollama-server-endpoint"

# For secure access to your Modal app
modal secret create MODAL_PROXY_TOKEN_ID "your-token-id"
modal secret create MODAL_PROXY_TOKEN_SECRET "your-token-secret"

# Optional - for restricted access to Gradio UI
modal secret create gradio_app_access_key "your-access-key"

# Required - Hugging Face token for accessing CSM model
modal secret create hf-secret "your-huggingface-token"
```

### Step 4: Create a Modal Volume and Deploy

```bash
modal volume create voice-chat-volume
modal deploy app_modal.py
```

## Using the Application

### Accessing the Web Interface

1. Navigate to the local URL or Modal deployment URL in your browser
2. For Modal, add `/ui` to the URL (e.g., `https://your-username--voice-chat-app-v1-serve.modal.run/ui`)

### Using the Voice Chat

1. **Select AI Service**: Choose between Scaleway (default) or Ollama
2. **Select a Model**: Choose from available models in the dropdown
   - Scaleway models: deepseek-r1-distill-llama-70b, meta-llama-3-70b-instruct, mixtral-8x7b-instruct-v0.1
   - Ollama models: mistral:latest, llama3:8b, llama3:70b, gemma3:27b, phi3:14b, mixtral:8x7b, codellama:70b
3. **Choose Voice Type**: Select "Woman" or "Man" for the AI's voice response
4. **Record Audio**: Click the microphone icon and speak your message
5. **Process Message**: Your speech will be automatically transcribed and processed
6. **Listen to Response**: The AI's response will be displayed as text and played as audio
7. **Reset Conversation**: Use the reset button to start a new conversation

## Speech Generation Features

The application uses two speech generation methods:

1. **CSM (Conditioned Sound Model)**:
   - High-quality voice synthesis for responses up to 300 characters
   - Consistent voice mapping (Man and Woman voices)
   - Optimized parameters for faster generation

2. **gTTS (Google Text-to-Speech)**:
   - Used for responses longer than 300 characters
   - Different voice mapping using regional accents:
     - Woman voice: US English (tld="us")
     - Man voice: UK English (tld="co.uk")

Text is sanitized before speech generation, with special handling for:
- Removal of asterisks and other problematic characters
- Preservation of standard punctuation for natural speech
- Normalization of quotes and whitespace

## Advanced Configuration Options

### Local Deployment Options

You can customize the application by modifying:

```bash
# Set the port for the Gradio interface
export GRADIO_PORT=8080

# Enable debug logging
export PYTHONPATH=./  # If needed to resolve import issues
```

### Making the App Public

To make your Gradio app publicly accessible:

1. With the local deployment, set `share=True` in the `demo.launch()` call (already configured)
2. For Modal deployment, the app is accessible via the provided URL

## Troubleshooting

### Common Issues

1. **Voice Mapping Confusion**: If voices sound incorrect, verify the CSM constants at the top of the file:
   ```python
   SPEAKER_ID_WOMAN = 0  # UI selection "Woman" 
   SPEAKER_ID_MAN = 1    # UI selection "Man"
   CSM_SPEAKER_WOMAN = 0  # CSM model expects ID 0 for woman
   CSM_SPEAKER_MAN = 1    # CSM model expects ID 1 for man
   ```

2. **Audio Not Working**: Ensure your browser has microphone permissions enabled

3. **LLM Connection Errors**:
   - For Scaleway: Verify your API key is correctly set
   - For Ollama: Verify your server URL is correct and accessible

4. **Missing gTTS**: If you see warnings about missing gTTS, install the packages:
   ```bash
   pip install gtts pydub
   ```

5. **Slow Speech Generation**: For responses close to the 300-character threshold, the CSM model may take time to generate audio. The application displays text immediately while audio is being prepared.

## Architecture

The application consists of the following components:

- **Whisper Model**: Converts speech to text
- **LLM Integration**:
  - Scaleway API for cloud-based LLMs
  - Ollama for self-hosted LLMs
- **CSM Model**: High-quality text-to-speech using Sesame's Conversational Speech Model
- **gTTS Fallback**: Faster alternative for long responses
- **Gradio UI**: Provides intuitive web interface

## License

This project is provided as-is for educational and demonstration purposes.

## Credits

Developed based on CSM (Conversational Speech Model) from [Sesame](https://www.sesame.com), which generates high-quality speech from text. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.
