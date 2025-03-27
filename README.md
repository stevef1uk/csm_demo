# Multimodal Voice Chat with Ollama LLM

This application provides a web-based voice chat interface powered by Ollama language models, deployed on Modal. It features speech-to-text, LLM processing, and text-to-speech capabilities in a single container.

## Features

- **Speech-to-Text**: Uses Whisper model to transcribe user voice input
- **LLM Integration**: Connects to remote Ollama server for text processing
- **Text-to-Speech**: Generates natural-sounding speech responses
- **Conversation Memory**: Maintains context throughout the conversation
- **GPU Acceleration**: Optimized for ML tasks using Modal's GPU capabilities
- **Secure Deployment**: Includes authentication for the Modal proxy

## Prerequisites

- A [Modal](https://modal.com/) account
- An Ollama server endpoint for LLM access
- Python 3.8+
- Modal CLI installed (`pip install modal`)

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

### Using the Voice Chat

1. **Select a Model**: Choose from available Ollama models in the dropdown
2. **Choose Voice Type**: Select "Man" or "Woman" for the AI's voice response
3. **Record Audio**: Click the microphone icon and speak your message
4. **Process Message**: Your speech will be automatically transcribed and processed
5. **Listen to Response**: The AI's response will be displayed as text and played as audio
6. **Reset Conversation**: Use the reset button to start a new conversation

### API Endpoints

The application also provides REST API endpoints:

- `POST /api/process_voice`: Process voice input and get response
- `POST /api/reset_conversation`: Reset the conversation history
- `GET /health`: Check if the API is running

## Customization Options

### GPU Configuration

You can customize the GPU by setting environment variables:

```bash
# Set GPU type and count
modal app update voice-chat-app-v1 --env GPU_TYPE=A100-80GB --env GPU_COUNT=1

# Set idle timeout (in seconds)
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

## Troubleshooting

### Common Issues

1. **Audio Not Working**: Ensure your browser has microphone permissions enabled
2. **LLM Connection Errors**: Verify your Ollama server URL is correct and accessible
3. **Modal Deployment Failures**: Check logs using `modal app logs voice-chat-app-v1`
4. **Authentication Issues**: Ensure your Modal proxy tokens are correctly set up

### Viewing Logs

```bash
modal app logs voice-chat-app-v1
```

## Architecture

The application consists of the following components:

- **Modal App**: Deploys the FastAPI and Gradio interfaces
- **Whisper Model**: Converts speech to text
- **Ollama Integration**: Communicates with the LLM server
- **Text-to-Speech**: Converts text responses to audio
- **Modal Volume**: Provides persistent storage for models and audio files

## License

This project is provided as-is for educational and demonstration purposes.

## Credits

Developed by Steven Fisher (stevef@gmail.com) based off CSM
CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

