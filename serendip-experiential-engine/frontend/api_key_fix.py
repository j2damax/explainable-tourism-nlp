"""
Direct OpenAI API key helper
This file directly provides the API key from a known source
"""
import os

def get_openai_api_key():
    """Get the OpenAI API key from various sources"""
    # First check environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        return api_key
        
    # Try to read from local .env file
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('OPENAI_API_KEY='):
                    # Extract the key, handling both quoted and unquoted values
                    key_value = line.strip().split('=', 1)[1]
                    if key_value.startswith('"') and key_value.endswith('"'):
                        key_value = key_value[1:-1]
                    return key_value
    
    # Try to read from parent directory .env file
    parent_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(parent_env_path):
        with open(parent_env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('OPENAI_API_KEY='):
                    # Extract the key, handling both quoted and unquoted values
                    key_value = line.strip().split('=', 1)[1]
                    if key_value.startswith('"') and key_value.endswith('"'):
                        key_value = key_value[1:-1]
                    return key_value
    
    # No key found
    return None