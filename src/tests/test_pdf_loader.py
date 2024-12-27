from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the NVIDIA API key
api_key = os.getenv('NVIDIA_API_KEY')
print(api_key)  # For testing; remove in production code
