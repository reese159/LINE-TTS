# LINE-TTS Developer Guide

## Overview (As seen in README.md)

LINE-TTS is a text-to-speech application written in python, providing the user the ability to:

- Generate their own voices for narration via blending of either open source or user-uploaded voice tensors
- Enter text or upload a PDF
- Generate a summary of the provided text with voice narration using an OpenAI model with configuration options for the model being used and the maximum number of tokens allowed in the summary
- Generate a full narration of the provided text
- Listen to or download either type of narration
- Download and save generated voice tensors for later use

## Technical Specs Overview

This project is built using Python 3.12, and leverages a lightweight, open-weight TTS model
(<https://huggingface.co/hexgrad/Kokoro-82M>) to create a Text-To-Speech (TTS) application with local desktop and deployed web application versions. The project makes use of libraries such as kokoro, pytorch,
streamlit, and Fitz (PyMuPDF). Tthe project is structured using one of two main entrypoint files found in the root directory of the repository that make use of several python modules found in the "mymodule" folder.

### Python modules

asdf

### Main Entrypoint Files

In plain terms, upon completion of these paths, a user should be able to use this application to
generate a narration of their submitted text using a narrator of their choosing or creation, with
the option to generate an additional summary of the text.

