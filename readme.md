# An AI assistant built with a tool-calling ensemble of LLMs  

This repository hosts an ai assistant powered by series of LLM Agents. It is designed to provide a human-like assistant with LLM tool calling that can be used through an OpenAI API type interface. This interface will allow this API to be used with many different LLM front ends. Built with LangChain and FastAPI.

## Key Features to be Implemented
* Extendable codebase to implement more tool-calling capabilities.
  * The most important tool will be web search through a SearX instance. In this manner, any chosen AI will have access to the latest information at any given time. 
* Freedom in LLM model selection. This will allow users to use cloud based models as well as local models.
* Experimentation with small LLM writers alongside larger LLM proof readers to generate messages for the user.
* Eventually the project will be Dockerized to allow a consistent and efficient way to install and host the API on remote computing platforms.

## Quick Start
```bash
git clone https://github.com/jSieber7/ai_assistant.git
cd ai_assistant
cp .env.template .env 

uv venv .venv
.venv\Scripts\activate # On Windows
uv sync --no-dev
python app/main.py
```