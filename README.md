# Multimodal Voice Chat with Ollama LLM

This application provides a web-based voice chat interface powered by Ollama language models, deployed on Modal. It features speech-to-text, LLM processing, and text-to-speech capabilities in a single container.

## Features

- **Speech-to-Text**: Uses Whisper model to transcribe user voice input
- **LLM Integration**: Connects to remote Ollama server for text processing
- **Text-to-Speech**: Generates natural-sounding speech with CSM (Conditioned Sound Model)
- **Conversation Memory**: Maintains context throughout the conversation
- **GPU Acceleration**: Optimized for ML tasks using Modal's GPU capabilities
- **Secure Deployment**: Includes authentication for the Modal proxy

## Prerequisites

- A [Modal](https://modal.com/) account
- An Ollama server endpoint for LLM access
- Python 3.8+
- Modal CLI installed (`pip install modal`)
- Hugging Face account (for CSM model access)

## Setup Instructions

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

### Step 4: Create a Modal Volume

```bash
modal volume create voice-chat-volume
```

## Deployment

### Step 1: Save the application code

Save the provided code as `app_modal.py`.

### Step 2: Deploy to Modal

```bash
modal deploy app_modal.py
```

After successful deployment, Modal will provide a URL where your application is hosted.

## Using the Application

### Accessing the Web Interface

1. Navigate to the provided Modal deployment URL in your browser
2. Add `/ui` to the URL to access the Gradio interface (e.g., `https://your-username--voice-chat-app-v1-serve.modal.run/ui`)
3. If you set a Gradio access key, you'll need to provide it
4. You will need to have a Browser extension installed on Chrome to add two Header keys for proxy auth protection e.g. Modal-Key & Modal-Secret values from the token created in Step 3
   I used Mod Header extension.

### Using the Voice Chat

1. **Select a Model**: Choose from available Ollama models in the dropdown
2. **Choose Voice Type**: Select "Woman" or "Man" for the AI's voice response
3. **Record Audio**: Click the microphone icon and speak your message
4. **Process Message**: Your speech will be automatically transcribed and processed
5. **Listen to Response**: The AI's response will be displayed as text and played as audio
6. **Reset Conversation**: Use the reset button to start a new conversation

### API Endpoints

The application also provides REST API endpoints:

- `POST /api/process_voice`: Process voice input and get response
- `POST /api/reset_conversation`: Reset the conversation history
- `GET /health`: Check if the API is running

## Speech Generation

The application uses two speech generation methods:

1. **CSM (Conditioned Sound Model)**: For high-quality voice synthesis of shorter responses 
2. **gTTS (Google Text-to-Speech)**: As a fallback and for longer responses to reduce latency

Speech generation is now optimized to:
- Display text responses immediately while audio is being generated
- Use gTTS for responses over 100 characters (for faster response time)
- Use CSM with optimized parameters for shorter responses where quality matters more

## Customization Options

### GPU Configuration

You can customize the GPU by setting environment variables:

```bash
# Set GPU type and count (default is H100)
modal app update voice-chat-app-v1 --env GPU_TYPE=A100-80GB --env GPU_COUNT=1

# Set idle timeout (in seconds, default is 90 seconds)
modal app update voice-chat-app-v1 --env IDLE_TIMEOUT=600
```

### Supported LLM Models

The application supports various Ollama models, including:
- llama3:8b
- llama3:70b
- gemma3:27b
- phi3:14b
- mistral:7b
- mixtral:8x7b
- codellama:70b

## Hugging Face Integration

The application uses Hugging Face models for:

1. **Speech recognition**: OpenAI Whisper model for transcription
2. **Speech synthesis**: Sesame CSM-1B model for high-quality voice generation

The Hugging Face authentication token is required to download these models, particularly the CSM model which requires authentication. The token is passed to the container via the `hf-secret` Modal secret.

## Troubleshooting

### Common Issues

1. **Audio Not Working**: Ensure your browser has microphone permissions enabled
2. **LLM Connection Errors**: Verify your Ollama server URL is correct and accessible
3. **Modal Deployment Failures**: Check logs using `modal app logs voice-chat-app-v1`
4. **Authentication Issues**: Ensure your Modal proxy tokens and Hugging Face token are correctly set up
5. **Slow Speech Generation**: For very long responses, the CSM model may take time to generate audio. The application now displays text immediately while audio is being prepared

### Viewing Logs

```bash
modal app logs voice-chat-app-v1
```

## Architecture

The application consists of the following components:

- **Modal App**: Deploys the FastAPI and Gradio interfaces
- **Whisper Model**: Converts speech to text
- **Ollama Integration**: Communicates with the LLM server
- **CSM Model**: High-quality text-to-speech using Sesame's Conversational Speech Model
- **gTTS Fallback**: Faster alternative for long responses
- **Modal Volume**: Provides persistent storage for models and audio files

## License

This project is provided as-is for educational and demonstration purposes.

## Credits

Developed by Steven Fisher (stevef@gmail.com) based on CSM.

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.
