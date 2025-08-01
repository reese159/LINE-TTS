# Loading openai api key
import os
from openai import OpenAI
from dotenv import load_dotenv

def summarize_text(text, model="gpt-3.5-turbo", max_tokens=250):
    """
    Summarizes the given text using OpenAI's GPT model.
    
    :text: The text to summarize.
    :model: The OpenAI model to use for summarization.
    :max_tokens: The maximum number of tokens in the summary.
    :return: The summarized text.
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


def summarize_text(text, model="gpt-3.5-turbo", max_tokens=250, openai_api_key=None):
    """
    Summarizes the given text using OpenAI's GPT model given user-supplied api key.
    
    :text: The text to summarize.
    :model: The OpenAI model to use for summarization.
    :max_tokens: The maximum number of tokens in the summary.
    :openai_api_key: The OpenAI API key to use for summarization.
    :return: The summarized text.
    """
    
    if openai_api_key != None:
        openai = OpenAI(api_key=openai_api_key)
    
        response = openai.chat.completions.create(
            model=model,
            messages = [
                {"role": "system", "content": "You are a helpful assistant for text summarization."},
                {"role": "user", "content": f"Please summarize the following text using at most {max_tokens} tokens:\n\n{text}"}
            ],
            max_tokens=max_tokens,  # Adjust as needed for summary length
        )
        return response.choices[0].message.content.strip()
    print("OpenAI API key is not provided. Please set the key to summarize text.")