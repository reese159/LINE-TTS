# Loading openai api key
import os
from openai import OpenAI
from dotenv import load_dotenv

def summarize_text(text, model="gpt-3.5-turbo", max_tokens=250):
    """
    Summarizes the given text using OpenAI's GPT model.
    
    Args:
        text (str): The text to summarize.
        model (str): The OpenAI model to use for summarization.
        max_tokens (int): The maximum number of tokens in the summary.
            Currently unused, could add for user flexibility in future.
        
    Returns:
        str: The summarized text.
    """
    
    load_dotenv()
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = openai.chat.completions.create(
        model=model,
        messages = [
            {"role": "system", "content": "You are a helpful assistant for text summarization."},
            {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
        ],
        max_tokens=max_tokens,  # Adjust as needed for summary length
    )
    return response.choices[0].message.content.strip()
