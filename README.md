# Medical Assistant API

This is a FastAPI application that provides a medical assistant API. The application uses LangChain agents, Google Generative AI, PubMed API, and FDA API to provide medical information.

## Features

- **Fetch PubMed Articles**: Retrieves research articles from PubMed based on user queries about symptoms, causes, and treatments.
- **Fetch Drug Side Effects**: Obtains detailed side effects of drugs from the FDA API.
- **Medical Assistant AI**: Provides concise, factual, and respectful responses to healthcare-related questions.

## Requirements

- **Python**: 3.8 or higher
- **Poetry**: For dependency management
- ## installation of required libraries 
- poetry add fastapi uvicorn pydantic requests python-dotenv langchain langchain-google-genai

## Create a .env file in the root directory and add your API keys:
    GOOGLE_API_KEY=your_google_api_key
    NCBI_API_KEY=your_pubmed_api_key
    FDA_API_KEY=your_fda_api_key



1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/medical-assistant-api.git
   cd medical-assistant-api
