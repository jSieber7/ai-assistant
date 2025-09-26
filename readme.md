# AI Assistant through a tool-calling ensemble of LLMs  

A series of LLM Agents designed to provide a human-like assistant with tool calling that can be used with an OpenAI API interface. Built with LangChain and FastAPI.

## Key Features to be Implemented
* Extendable codebase to implement more tool-calling capabilities.
  * The most important tool will be web search through a SearX instance. In this manner, any chosen AI will have access to the latest information at any given time. 
* Freedom in LLM model selection, from cloud based to local
* Experimentation with small LLM writers alongside larger LLM proof readers.
* Eventually the project will be Dockerized to allow a consistent and efficient way to install and host on remote computing platforms.

## Quick Start
Clone the repository
```bash
cd ai_assistant
cp .env.template .env 

uv venv .venv
.venv\Scripts\activate # On Windows
uv sync --no-dev
python app/main.py
```