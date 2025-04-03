# llm_services.py
import requests
import json
import logging
import time
import os
from openai import OpenAI

# Import from the correct module
from text_utils import sanitize_text_for_tts

# Set up logger
logger = logging.getLogger(__name__)

# Global variables
CURRENT_SERVICE = "Scaleway"  # Default service
SCALEWAY_API_KEY = ""

# Session management access
def get_conversation_history(session_id=None):
    from session_management import get_conversation_history
    return get_conversation_history(session_id)

def add_to_conversation(session_id, role, content):
    from session_management import add_to_conversation
    add_to_conversation(session_id, role, content)

def set_api_key(api_key):
    """Set the Scaleway API key"""
    global SCALEWAY_API_KEY
    SCALEWAY_API_KEY = api_key
    logger.info(f"Scaleway API key set: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")

def update_service_selection(service):
    """Update the global service selection variable"""
    global CURRENT_SERVICE
    old_service = CURRENT_SERVICE
    CURRENT_SERVICE = service
    print(f"🔄 Service changed from {old_service} to {CURRENT_SERVICE}")
    return service

def chat_with_ollama(message, model_name, ollama_url, session_id=None):
    """Send message to Ollama and get response"""
    conversation_history = get_conversation_history(session_id) if session_id else []
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
        end_time = time.time()
        api_time = end_time - api_start_time
        message = f"⏱️ Ollama API request took {api_time:.2f} seconds"
        logger.info(message)
        print(message)
        
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
                if session_id:
                    add_to_conversation(session_id, "user", message)
                    add_to_conversation(session_id, "assistant", response_text)
                
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

def chat_with_scaleway(message, model_name, api_key, session_id=None):
    """Send message to Scaleway LLM API and get response"""
    conversation_history = get_conversation_history(session_id) if session_id else []
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
            end_time = time.time()
            api_time = end_time - api_start_time
            message_log = f"⏱️ Scaleway API request took {api_time:.2f} seconds"
            logger.info(message_log)
            print(message_log)
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Sanitize the response text
            response_text = sanitize_text_for_tts(response_text)
            logger.info(f"Sanitized Scaleway response: '{response_text}'")
            
            # Update conversation history
            if session_id:
                add_to_conversation(session_id, "user", message)
                add_to_conversation(session_id, "assistant", response_text)
            
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

def get_llm_response(transcription, model_name, ollama_url, service=None, session_id=None):
    """Get LLM response only, without waiting for audio generation"""
    global CURRENT_SERVICE, SCALEWAY_API_KEY
    
    if not transcription or transcription == "":
        return "No text to process", ""
    
    try:
        # Start timing for LLM response
        llm_start_time = time.time()
        
        # Determine which service to use with fallback options
        current_service = service if service else CURRENT_SERVICE
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
            response = chat_with_scaleway(transcription, model_name, SCALEWAY_API_KEY, session_id)
        else:  # Ollama
            print(f"🔶 Attempting to connect to Ollama at: {ollama_url}")
            response = chat_with_ollama(transcription, model_name, ollama_url, session_id)
        
        # Log time taken for LLM response
        end_time = time.time()
        llm_elapsed = end_time - llm_start_time
        log_message = f"⏱️ LLM response took {llm_elapsed:.2f} seconds"
        logger.info(log_message)
        print(log_message)
        
        # Return text response immediately
        return f"LLM response received ({llm_elapsed:.2f}s), generating audio...", response
            
    except Exception as e:
        logger.error(f"Error processing with LLM: {str(e)}")
        logger.exception(e)
        return f"Error: {str(e)}", ""

def toggle_service_options(service):
    """Handle service toggling between Scaleway and Ollama"""
    global CURRENT_SERVICE
    CURRENT_SERVICE = service  # Update global variable
    
    print(f"📢 Service toggled to: {service}")
    print(f"Global CURRENT_SERVICE is now: {CURRENT_SERVICE}")
    
    import gradio as gr
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
