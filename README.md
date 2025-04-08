# Multimodal Voice Chat with AI Models

This application provides a voice chat interface powered by AI language models (Scaleway and Ollama). It features speech-to-text, LLM processing, and text-to-speech capabilities with support for multiple voice types and session management for saving conversations.

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
- **Session Management**: Save, load, and manage conversation sessions with unique identifiers
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

### For Scaleway Serverless Container Deployment

- Scaleway account with Container Registry and Serverless Containers enabled
- Docker installed locally
- Scaleway CLI (optional)
- Container Registry namespace

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

# Optional session management configuration
export SESSION_DIR="user_sessions"  # Directory to store session files
export SESSION_RETENTION_DAYS="30"  # How long to keep session files (default: 30 days)
export CLEANUP_INTERVAL_HOURS="24"  # How often to check for old sessions (default: 24 hours)
```

### Step 4: Run the application

```bash
python app_scaleway.py
```

The application will be available at http://localhost:7860 by default.

## Setup Instructions for Scaleway Serverless Container Deployment

### Step 1: Create a Dockerfile

Create a file named `Dockerfile` with the following content:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Copy application file
COPY app_simple_scaleway.py ./app.py

# Install packages in separate steps for better reliability
RUN pip install --no-cache-dir \
    gradio==5.23.1 \
    transformers==4.35.2 \
    numpy==1.25.2 \
    requests==2.31.0 \
    gtts==2.3.2 \
    pydub==0.25.1 \
    soundfile==0.12.1 \
    PyAudio==0.2.13

# Install PyTorch separately with its custom index
RUN pip install --no-cache-dir \
    torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu

# Pre-download models to avoid runtime downloads
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    processor = WhisperProcessor.from_pretrained('openai/whisper-base'); \
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')"

# Create directories
RUN mkdir -p audio_outputs user_sessions

# Expose port
EXPOSE 7860

# Start application
CMD ["python", "app.py"]
```

### Step 2: Build the Docker image for AMD64 architecture

```bash
docker buildx build --load --platform=linux/amd64 -t scaleway-voice-chat .
```

### Step 3: Push to Scaleway Container Registry

```bash
# Login to Scaleway Container Registry (replace with your namespace)
docker login rg.nl-ams.scw.cloud/your-namespace -u nologin -p your-scaleway-api-key

# Tag the image
docker tag scaleway-voice-chat rg.nl-ams.scw.cloud/your-namespace/scaleway-voice-chat:latest

# Push to registry
docker push rg.nl-ams.scw.cloud/your-namespace/scaleway-voice-chat:latest
```

### Step 4: Deploy as Serverless Container

**Option 1: Using Scaleway Console**

1. Go to the [Scaleway Console](https://console.scaleway.com/)
2. Navigate to "Serverless" → "Containers"
3. Click "Create Container"
4. Configure your container:
   - Name: `scaleway-voice-chat`
   - Container Image: Select your registry and `scaleway-voice-chat:latest` image
   - Memory: 2GB
   - CPU: 2 vCPU
   - Environment Variables:
     - Name: `SCALEWAY_API_KEY`
     - Value: Your Scaleway API key
     - Type: Secret (ensures it's stored securely)
   - Port: 7860
5. Click "Create Container"

**Option 2: Using Scaleway CLI**

```bash
# Install Scaleway CLI if needed
curl -o /usr/local/bin/scw -L "https://github.com/scaleway/scaleway-cli/releases/latest/download/scw-darwin-arm64"
chmod +x /usr/local/bin/scw
scw init

# Deploy container
scw container namespace function create \
  --namespace-id your-namespace \
  --name scaleway-voice-chat \
  --registry-image rg.nl-ams.scw.cloud/your-namespace/scaleway-voice-chat:latest \
  --memory-limit 2G \
  --port 7860 \
  --env SCALEWAY_API_KEY=your-api-key:secret \
  --region nl-ams
```

### Step 5: Access Your Deployed Application

Once deployed, you can access your application at the URL provided by Scaleway Serverless Containers.

## Using the Application

### Accessing the Web Interface

1. Navigate to the local URL, Modal deployment URL, or Scaleway Serverless Container URL in your browser
2. For Modal, add `/ui` to the URL (e.g., `https://your-username--voice-chat-app-v1-serve.modal.run/ui`)
3. For Scaleway Serverless Container, use the URL provided in the console (e.g., `https://your-function-id.functions.fnc.fr-par.scw.cloud`)

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

### Using Session Management

The application includes session management features that allow you to save and reload your conversations:

1. **Session ID**: Each conversation is associated with a unique session ID, which is randomly generated by default
2. **Custom Session ID**: You can create a memorable name for your session using the "Custom Session ID" field
   - Enter a name (e.g., "work-chat" or "medical-questions")
   - Click "Apply Custom ID" to assign this name to your session
3. **Save Session**: Click the "💾 Save Session" button to store your current conversation
4. **Load Session**: Click the "📂 Load Session" button to reload a previously saved conversation
5. **Clear Session**: Click the "🗑️ Clear Session" button to start fresh while keeping the same session ID

Sessions are isolated between different browser windows and tabs, allowing you to maintain multiple separate conversations simultaneously. Each session has its own conversation history that persists between page refreshes when saved.

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

# Session management configuration
export SESSION_DIR="user_sessions"  # Directory where session files are stored
export SESSION_RETENTION_DAYS="30"  # Days to keep session files before automatic deletion
export CLEANUP_INTERVAL_HOURS="24"  # How often to run the cleanup job (in hours)

# Enable debug logging
export PYTHONPATH=./  # If needed to resolve import issues
```

### Scaleway Serverless Container Options

When deploying to Scaleway Serverless Containers, you can customize:

1. **Environment Variables**:
   - `SCALEWAY_API_KEY`: Your Scaleway API key (set as a secret)
   - `SESSION_DIR`: Directory where session files are stored
   - `SESSION_RETENTION_DAYS`: Days to keep session files
   - `CLEANUP_INTERVAL_HOURS`: How often to run cleanup

2. **Resources**:
   - Memory: 2GB recommended for good performance
   - vCPU: 2 vCPU recommended for faster processing
   - Minimum Scale: 0 (scale to zero when not in use)
   - Maximum Scale: Based on your expected traffic

3. **Security**:
   - Always set API keys as Secrets in the console
   - Consider adding authentication if needed
   - Enable HTTPS termination (recommended)

### Making the App Public

To make your Gradio app publicly accessible:

1. With the local deployment, set `share=True` in the `demo.launch()` call (already configured)
2. For Modal deployment, the app is accessible via the provided URL
3. For Scaleway Serverless Container, the app is automatically accessible via the provided URL

### Session Storage

Session data is stored in JSON files within the `SESSION_DIR` directory (defaults to "user_sessions"). Each session file contains:
- A hashed session ID for privacy
- The conversation history (user and AI messages)
- The last updated timestamp

The system automatically cleans up old session files based on the `SESSION_RETENTION_DAYS` setting.

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

6. **Session Management Issues**:
   - If session loading fails, check if the session file exists in the SESSION_DIR directory
   - If custom IDs don't work, ensure there are no special characters in your ID
   - If sessions are shared between browsers, ensure you're using a different Custom Session ID for each

7. **Serverless Container Issues**:
   - Container not starting: Check for logs in the Scaleway console
   - Long cold start times: Pre-downloading models in the Dockerfile helps
   - Memory errors: Increase the memory allocation
   - Microphone not working: HTTPS is required for microphone access, ensure it's enabled

## Architecture

The application consists of the following components:

- **Whisper Model**: Converts speech to text
- **LLM Integration**:
  - Scaleway API for cloud-based LLMs
  - Ollama for self-hosted LLMs
- **CSM Model**: High-quality text-to-speech using Sesame's Conversational Speech Model
- **gTTS Fallback**: Faster alternative for long responses
- **Session Management**: Stores and manages conversation histories
- **Gradio UI**: Provides intuitive web interface

The application is structured with modular architecture:
- `app_scaleway.py`: Main application file for local deployment
- `app_scaleway_modal.py`: Version optimized for Modal deployment
- `app_simple_scaleway.py`: Simplified version for Scaleway Serverless Containers
- `audio_utils.py`: Audio processing and transcription
- `llm_services.py`: Connects to language models
- `text_utils.py`: Text processing and formatting
- `tts_service.py`: Text-to-speech functionality
- `session_management.py`: Session storage and retrieval

## License

This project is provided as-is for educational and demonstration purposes.

## Credits

Developed based on CSM (Conversational Speech Model) from [Sesame](https://www.sesame.com), which generates high-quality speech from text. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.
