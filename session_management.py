import os
import hashlib
import threading
import secrets
import json
import datetime
import logging

# Logger setup
logger = logging.getLogger(__name__)

# Configuration for session cleanup
SESSION_DIR = os.getenv("SESSION_DIR", "user_sessions")
SESSION_RETENTION_DAYS = int(os.getenv("SESSION_RETENTION_DAYS", "30"))  # Default: 30 days
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))  # Default: Once daily

# Store conversation histories by session ID
session_conversations = {}

def get_session_path(user_id):
    """Get the file path for a user's session file"""
    # Create a safe filename from the user ID
    safe_id = hashlib.sha256(user_id.encode()).hexdigest()
    return os.path.join(SESSION_DIR, f"session_{safe_id}.json")

def save_user_session(user_id):
    """Save the current conversation for a user"""
    global session_conversations
    
    if not user_id or user_id == "None":
        logger.warning("No valid user ID provided for session saving")
        return "No user ID, session not saved"
    
    try:
        # Get the conversation for this session
        conversation_history = session_conversations.get(user_id, [])
        
        # Create session data structure
        session_data = {
            "user_id": user_id,
            "last_updated": datetime.datetime.now().isoformat(),
            "conversation_history": conversation_history
        }
        
        # Save to file
        session_path = get_session_path(user_id)
        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        logger.info(f"Saved session for user {user_id[:8]}... with {len(conversation_history)} messages")
        return f"Session saved: {len(conversation_history)} messages"
    
    except Exception as e:
        logger.error(f"Error saving user session: {str(e)}")
        return f"Error saving session: {str(e)}"

def load_user_session(user_id):
    """Load a user's conversation history"""
    global session_conversations
    
    if not user_id or user_id == "None":
        logger.warning("No valid user ID provided for session loading")
        return "No user ID provided", ""
    
    try:
        session_path = get_session_path(user_id)
        
        # Check if session file exists
        if not os.path.exists(session_path):
            logger.info(f"No existing session for user {user_id[:8]}...")
            session_conversations[user_id] = []
            return "New session created", ""
        
        # Load the session file
        with open(session_path, 'r') as f:
            session_data = json.load(f)
            
        # Update conversation history for this session ID
        conversation_history = session_data.get("conversation_history", [])
        session_conversations[user_id] = conversation_history
        
        # Generate display text
        display_text = ""
        for msg in conversation_history:
            role = "You" if msg["role"] == "user" else "AI"
            display_text += f"{role}: {msg['content']}\n\n"
        
        # Log the session restoration
        msg_count = len(conversation_history)
        last_updated = session_data.get("last_updated", "unknown")
        logger.info(f"Loaded session for user {user_id[:8]}... with {msg_count} messages from {last_updated}")
        
        return f"Loaded session with {msg_count} messages", display_text
    
    except Exception as e:
        logger.error(f"Error loading user session: {str(e)}")
        return f"Error loading session: {str(e)}", ""

def clear_user_session(user_id):
    """Clear a user's conversation history but maintain the user ID"""
    global session_conversations
    
    if not user_id or user_id == "None":
        return "No user ID, nothing to clear"
    
    try:
        # Reset conversation history for this session
        session_conversations[user_id] = []
        
        # Update the session file with empty conversation
        save_user_session(user_id)
        
        logger.info(f"Cleared session for user {user_id[:8]}...")
        return "Conversation cleared"
    
    except Exception as e:
        logger.error(f"Error clearing user session: {str(e)}")
        return f"Error clearing session: {str(e)}"

def cleanup_old_sessions():
    """Remove session files older than the retention period"""
    import time  # Import here to avoid circular imports
    
    try:
        logger.info(f"Starting session cleanup job (retention: {SESSION_RETENTION_DAYS} days)")
        
        # Calculate the cutoff timestamp (current time - retention period)
        cutoff_time = time.time() - (SESSION_RETENTION_DAYS * 86400)  # 86400 seconds in a day
        
        # Get all session files
        session_files = os.listdir(SESSION_DIR)
        cleaned_count = 0
        
        for filename in session_files:
            if not filename.startswith("session_") or not filename.endswith(".json"):
                continue  # Skip non-session files
                
            filepath = os.path.join(SESSION_DIR, filename)
            
            # Check file modification time
            file_mtime = os.path.getmtime(filepath)
            
            # If file is older than retention period, delete it
            if file_mtime < cutoff_time:
                try:
                    os.remove(filepath)
                    cleaned_count += 1
                    logger.debug(f"Deleted old session file: {filename}")
                except Exception as e:
                    logger.error(f"Error deleting session file {filename}: {str(e)}")
        
        logger.info(f"Session cleanup completed: removed {cleaned_count} sessions older than {SESSION_RETENTION_DAYS} days")
        
        # Schedule the next cleanup
        schedule_next_cleanup()
        
    except Exception as e:
        logger.error(f"Error in session cleanup job: {str(e)}")
        # Try to schedule next cleanup despite error
        schedule_next_cleanup()

def schedule_next_cleanup():
    """Schedule the next cleanup job"""
    import time  # Import here to avoid circular imports
    
    # Convert hours to seconds
    interval_seconds = CLEANUP_INTERVAL_HOURS * 3600
    
    cleanup_thread = threading.Timer(interval_seconds, cleanup_old_sessions)
    cleanup_thread.daemon = True  # Allow the program to exit even if thread is running
    cleanup_thread.start()
    
    logger.info(f"Next session cleanup scheduled in {CLEANUP_INTERVAL_HOURS} hours")

def generate_fallback_session_id():
    """Generate a random session ID for users without JavaScript"""
    # Create a random 16-byte token and convert to hex
    return secrets.token_hex(16)

def apply_custom_session_id(custom_id):
    """Allow users to set their own session ID"""
    global session_conversations
    
    if not custom_id or custom_id.strip() == "":
        return "No custom ID provided", "Please enter a valid ID"
    
    # Hash the custom ID for privacy and consistency
    hashed_id = hashlib.sha256(custom_id.encode()).hexdigest()
    
    # Initialize this session if it doesn't exist
    if hashed_id not in session_conversations:
        session_conversations[hashed_id] = []
        # Try to load from file if it exists
        session_path = get_session_path(hashed_id)
        if os.path.exists(session_path):
            try:
                with open(session_path, 'r') as f:
                    session_data = json.load(f)
                session_conversations[hashed_id] = session_data.get("conversation_history", [])
                msg_count = len(session_conversations[hashed_id])
                logger.info(f"Loaded existing session for custom ID {custom_id} with {msg_count} messages")
            except Exception as e:
                logger.error(f"Failed to load existing session for custom ID: {e}")
    
    return hashed_id, f"Custom session ID applied: {custom_id[:8]}... (hashed for privacy)"

def initialize_session_management():
    """Initialize the session directory and cleanup job"""
    # Ensure the session directory exists
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    # Log the session configuration
    logger.info(f"Session storage directory: {os.path.abspath(SESSION_DIR)}")
    logger.info(f"Session retention period: {SESSION_RETENTION_DAYS} days")
    logger.info(f"Cleanup interval: {CLEANUP_INTERVAL_HOURS} hours")
    
    # Start the cleanup job with a small delay
    initial_delay = 120  # 2 minutes
    first_cleanup = threading.Timer(initial_delay, cleanup_old_sessions)
    first_cleanup.daemon = True
    first_cleanup.start()
    
    logger.info(f"Session cleanup job scheduled (first run in {initial_delay} seconds)")

# Function to get the conversation history for a specific session
def get_conversation_history(session_id=None):
    global session_conversations
    
    if not session_id:
        # No session specified, return empty history
        return []
        
    # Return the history for the specified session, or empty list if not found
    if session_id not in session_conversations:
        session_conversations[session_id] = []
        
    return session_conversations.get(session_id, [])

# Function to set the conversation history for a specific session
def set_conversation_history(session_id, history):
    global session_conversations
    
    if not session_id:
        # No session specified, do nothing
        return
    
    # Set the history for the specified session
    session_conversations[session_id] = history

# Function to add a message to a specific session's conversation history
def add_to_conversation(session_id, role, content):
    global session_conversations
    
    if not session_id:
        # No session specified, do nothing
        return
    
    # Initialize the session if it doesn't exist
    if session_id not in session_conversations:
        session_conversations[session_id] = []
    
    # Add the message to the conversation
    session_conversations[session_id].append({"role": role, "content": content})
