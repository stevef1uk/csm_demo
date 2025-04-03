import re
import logging

# Logger setup
logger = logging.getLogger(__name__)

def sanitize_text_for_tts(text):
    """Clean up text for TTS without removing standard punctuation"""
    if text is None:
        return ""
    
    # Remove emojis and other unicode special characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove control characters and non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    # Replace special quotes with standard quotes
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    
    # SPECIFICALLY REMOVE ASTERISKS AND EXCLAMATION POINTS 
    text = text.replace('*', '')
    text = text.replace('!', ' ')  # Replace exclamation marks with spaces
    
    # LESS AGGRESSIVE FILTERING: Allow common punctuation and symbols
    # Only remove truly problematic characters
    text = re.sub(r'[^\w\s.,?()\'":;\-–—+&%$#@/\\|{}\[\]<>]', '', text)
    
    # Normalize whitespace (replace multiple spaces with a single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Only add ending punctuation if there's none already
    if text and not re.search(r'[.?]$', text):
        text += '.'
    
    return text

def update_conversation_display(conversation_history):
    """Update the conversation history display"""
    if not conversation_history:
        return ""
    
    display_text = ""
    for msg in conversation_history:
        role = "You" if msg["role"] == "user" else "AI"
        display_text += f"{role}: {msg['content']}\n\n"
    
    return display_text

def log_timing(operation_name, start_time, logger):
    """Log the time taken for an operation"""
    import time
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    message = f"⏱️ {operation_name} took {elapsed_seconds:.2f} seconds"
    logger.info(message)
    print(message)
    return elapsed_seconds
