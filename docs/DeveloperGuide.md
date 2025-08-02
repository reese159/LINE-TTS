# LINE-TTS Developer Guide

## Overview

LINE-TTS is a text-to-speech application written in python, providing the user the ability to:

- Generate their own voices for narration via blending of either open source or user-uploaded voice tensors
- Enter text or upload a PDF
- Generate a summary of the provided text with voice narration using an OpenAI model with configuration options for the model being used and the maximum number of tokens allowed in the summary
- Generate a full narration of the provided text
- Listen to or download either type of narration
- Download and save generated voice tensors for later use

In plain terms, this application allows a user to generate a narration of their submitted text using a narrator of their choosing or creation, with the option to generate an additional summary of the text.

## Local Setup

This project was created using python 3.12, please install to run locally.

All requirements can be found in "requirements.txt", found in the root of this repository. Can be installed directly to a virtual environment by running "pip install -r requirements.txt" in the terminal with the venv active.

After installation, the user can run the local version of this application by opening a terminal in the root directory of the project with the venv activated, and entering the following command:

streamlit run local_streamlit_narrator.py

## *Optional*

For text summarization to funtion locally, the user will need to generate an api key from OpenAI, which can be purchased on the OpenAI Platform. After obtaining the API key, the user will need to generate a "secrets.toml" file in the root directoy of the project, containing the OpenAI API key in the following format: OPENAI_API_KEY="your_api_key_here"

## Technical Specs Overview

This project is built using Python 3.12, and leverages a lightweight, open-weight TTS model
(<https://huggingface.co/hexgrad/Kokoro-82M>) to create a Text-To-Speech (TTS) application with local desktop and deployed web application versions. The project makes use of libraries such as kokoro, pytorch,
streamlit, and Fitz (PyMuPDF). Tthe project is structured using one of two main entrypoint files found in the root directory of the repository that make use of several python modules found in the "mymodule" folder. The python files that should concern a developer exist in the following structure, and are expanded on per folder:

```text
LINE-TTS/
│   ├──mymodule/
│   │   ├── __init__.py
│   │   ├── audio_joiner.py
│   │   ├── file_reader.py
│   │   ├── text_summarization.py
│   │   └── voice_blend.py
├── deployed_streamlit_narrator.py
└── local_streamlit_narrator.py
```

### Python modules

The project's core logic is encapsulated within the `mymodule` folder. Below is a diagram of its structure and a description of each module's purpose.

```text
├──mymodule/
│   ├── __init__.py
│   ├── audio_joiner.py
│   ├── file_reader.py
│   ├── text_summarization.py
│   └── voice_blend.py
```

- **`audio_joiner.py`**: This module provides utilities for audio manipulation. It includes functions to concatenate audio segments and to convert audio tensors from the TTS model into a playable audio format.

- **`file_reader.py`**: Responsible for handling file inputs, specifically for extracting text content from PDF files. It includes logic to crop headers and footers to get clean text for narration.

- **`text_summarization.py`**: This module interfaces with the OpenAI API to provide text summarization capabilities. It takes a body of text and returns a condensed version.

- **`voice_blend.py`**: The core of the voice customization feature. This module contains the logic for blending multiple voice tensors based on user-defined weights to create a unique, custom voice for narration.

To ensure `mymodule` is treated as a proper Python package, it's good practice to include an `__init__.py` file, even if it's empty.

### Main Entrypoint Files

```text
LINE-TTS/
├── deployed_streamlit_narrator.py
└── local_streamlit_narrator.py
```

- **`deployed_streamlit_narrator.py`**: This module integrates the modules into a cohesive user interface leveraging the Streamlit to create and deploy a website on streamlit's community cloud service.
- **`local_streamlit_narrator.py`**: This module integrates the modules into a cohesive user interface leveraging the Streamlit for the user to run entirely locally.

## User Flow Walkthrough

Below is a technical walkthrough of the local version of the application, using the **`local_streamlit_narrator.py`** entrypoint file as a guide. Note: the only noteworthy differences in implementation between this version and the deployed version should be a maximum number of characters allowed for narration in the deployed version, and the addition of an option for a "secrets.toml" file to be included in the root directory fo the local version.
TODO: ADD USER FLOW DIAGRAM

## Known Issues

### Major

- The browser version of this application will crash from time to time due to the application going over its resource limits as provided by Streamlit's Community Cloud, using too much memory.

### Minor

- No fallback implemented for the local OpenAI text summarization method.

## Future Work

- Developed using OpenAI API documentation and model current as of 08/02/2025, could be updated to utilize a legacy implementiation as a fallback or a method not requiring an API key to allow a user to access the text summarization feature without requriing an API key.
- Potential integration of caching mechanisms offered by Streamlit (e.g. @st.cache_data and @st.cache_resource) to optimize resouce management, speed application responsiveness, and lower memory constraints.
- Separating more reused code into functions - some processes, e.g. those found in summary narration and full narration, have components that are repeated. For cleanliness, segments could be constrained to a "generate_narration" function.
