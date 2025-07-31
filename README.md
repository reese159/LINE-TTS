# LINE-TTS

Local Interactive Narration Environment for Text-To-Speech

---

## Introduction

This project contains both an online and local version, with the online version of the application existing primarily as a showcase of the application's capabilities with hard limits on the length of narrations that can be generated to avoid exceeding streamlit's resource limits. As such, for full use of narration generation and maximum user privacy, the local version is recommended.

Aside from the character limit and OpenAI API key configuration (should the user wish to generate text summaries in the local version), the functionality of both versions of the application should be identical.

Below is a typical user workflow:

Step 1, select/upload voices:

![User_Flow_Recording_1-Voice_Selection](https://github.com/user-attachments/assets/97bfe74c-90c7-4315-99b8-640a532c6c14)

Step 2, set voice weights:

![User_Flow_Recording_2-Weight_Selection](https://github.com/user-attachments/assets/4beeab93-7d60-4b4f-9ab6-7732ee5b1b1e)

Step 3, provide text input for summary/narration:

![User_Flow_Recording_3-text_input](https://github.com/user-attachments/assets/90e7d21b-ade4-42dc-a3be-8b3b8e477e23)

![User_Flow_Recording_3-pdf_input](https://github.com/user-attachments/assets/151a93f8-a4ed-4f73-9d0e-fdc38e8ad043)

Step 4, generate summary:

![User_Flow_Recording_4-text_summary_generation](https://github.com/user-attachments/assets/e7de7d32-a258-4c45-9778-3e1fe8dfe663)

Step 5, generate full narration:

![User_Flow_Recording_5-narration_generation](https://github.com/user-attachments/assets/0ac60673-2df8-4487-8d95-5a2922b49874)

Note, after either step 4 or 5, the user can download both the narration as well as the blended voice itself for future use.

## Local Setup

This project was created using [python 3.12](https://www.python.org/downloads/release/python-31210/), please install to run locally.

All requirements can be found in "requirements.txt", found in the root of this repository. Can be installed directly to a virtual environment by running "pip install -r requirements.txt" in the terminal with the venv active.

After installation, the user can run the local version of this application by opening a terminal in the root directory of the project with the venv activated, and entering the following command:

streamlit run local_streamlit_narrator.py

### *Optional*

For text summarization to funtion locally, the user will need to generate an api key from OpenAI, which can be purchased on the [OpenAI Platform](https://platform.openai.com/settings/organization/api-keys). After obtaining the API key, the user will need to generate a "secrets.toml" file in the root directoy of the project, containing the OpenAI API key in the following format:
OPENAI_API_KEY="your_api_key_here"

---
#### Credits
All open-weight models provided can be found on huggingface under [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) created by [hexgrad](https://huggingface.co/hexgrad) under the [Apache 2.0 License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md). 
Direct download for voice-tensor files provided can be found under [voices](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices) in the Kokoro-82M repository. 
Credit to [OpenAI](https://platform.openai.com/docs/overview) for the provided models used in text summarization.
