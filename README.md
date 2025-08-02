# LINE-TTS

Local Interactive Narration Environment for Text-To-Speech

---

## Introduction

LINE-TTS is a text-to-speech application written in python, providing the user the ability to:

- Generate their own voices for narration via blending of either open source or user-uploaded voice tensors
- Enter text or upload a PDF
- Generate a summary of the provided text with voice narration using an OpenAI model with configuration options for the model being used and the maximum number of tokens allowed in the summary
- Generate a full narration of the provided text
- Listen to or download either type of narration
- Download and save generated voice tensors for later use

This project contains both an online and local version, with the online version of the application existing primarily as a showcase of the application's capabilities with hard limits on the length of narrations that can be generated to avoid exceeding streamlit's resource limits. As such, for full use of narration generation and maximum user privacy, the local version is recommended.

The browser version of this application can be found at [https://linetts.streamlit.app/](https://linetts.streamlit.app/)

Aside from the character limit and OpenAI API key configuration (should the user wish to generate text summaries in the local version), the functionality of both versions of the application should be identical.

Below is a typical user workflow:

Step 1, select/upload voices:

![User_Flow_Recording_1-Voice_Selection](https://github.com/user-attachments/assets/ea78a34b-7cea-43a1-a364-468ccd565f7a)

Step 2, set voice weights:

![User_Flow_Recording_2-Weight_Selection](https://github.com/user-attachments/assets/0e4563d6-2b15-4103-bf41-87ae0f71721e)

Note, the user should input weights summing to 1.0 to avoid warnings.

Step 3, provide text input for summary/narration using the text box provided or uploadinga valid PDF:

![User_Flow_Recording_3-text_input](https://github.com/user-attachments/assets/52156dd4-82ae-4325-bf43-c2be7fba808c)

![User_Flow_Recording_3-pdf_input](https://github.com/user-attachments/assets/3b753326-ef66-42cb-8ef6-d53522d240e1)

Step 4, generate summary or full narration:

![User_Flow_Recording_4-text_summary_generation](https://github.com/user-attachments/assets/9402a8d3-6eb5-4773-a320-de2503259b54)

![User_Flow_Recording_4-narration_generation](https://github.com/user-attachments/assets/86ecff5f-5404-48ac-b621-c1f00200d263)


Note, after either step 4, the user can download both the narration as well as the blended voice itself for future use.

Obviously, not all stops are necessary, as the user may only want ot benerate a summary or full narration, with the summary being heavily recommended in the browser version of the application.

## Local Setup

This project was created using [python 3.12](https://www.python.org/downloads/release/python-31210/), please install to run locally.

All requirements can be found in "requirements.txt", found in the root of this repository. Can be installed directly to a virtual environment by running "pip install -r requirements.txt" in the terminal with the venv active.

After installation, the user can run the local version of this application by opening a terminal in the root directory of the project with the venv activated, and entering the following command:

streamlit run local_streamlit_narrator.py

## *Optional*

For text summarization to funtion locally, the user will need to generate an api key from OpenAI, which can be purchased on the [OpenAI Platform](https://platform.openai.com/settings/organization/api-keys). After obtaining the API key, the user will need to generate a "secrets.toml" file in the root directoy of the project, containing the OpenAI API key in the following format:
OPENAI_API_KEY="your_api_key_here"

---

### Credits

All open-weight models provided can be found on huggingface under [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) created by [hexgrad](https://huggingface.co/hexgrad) under the [Apache 2.0 License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).

Direct download for voice-tensor files provided can be found under [voices](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices) in the Kokoro-82M repository.

Credit to [OpenAI](https://platform.openai.com/docs/overview) for the provided models used in text summarization.
