from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import uvicorn

# Load environment variables
load_dotenv()

# API keys
PUBMED_API_KEY = os.getenv("NCBI_API_KEY")
FDA_API_KEY = os.getenv("FDA_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Initialize the LLM with GoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Function to query PubMed for symptoms and treatment options
def fetch_pubmed_articles(search_term: str) -> str:
    """Fetches articles from PubMed based on the provided search term."""
    PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_SUMMARY_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        'db': 'pubmed',
        'term': search_term,
        'retmax': 5,
        'retmode': 'json',
        'api_key': PUBMED_API_KEY
    }

    try:
        # Search PubMed
        response = requests.get(PUBMED_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return "No articles found for this search term."
            
            # Fetch summaries for the retrieved articles
            summary_params = {
                'db': 'pubmed',
                'id': ','.join(id_list),
                'retmode': 'json',
                'api_key': PUBMED_API_KEY
            }
            summary_response = requests.get(PUBMED_SUMMARY_API_URL, params=summary_params)
            if summary_response.status_code == 200:
                summary_data = summary_response.json()
                articles = summary_data.get("result", {})
                detailed_responses = []
                for article_id in id_list:
                    article = articles.get(article_id, {})
                    title = article.get("title", "No title available")
                    pub_year = article.get("pubdate", "No publication date available")
                    detailed_responses.append(f"Title: {title}, Published: {pub_year}")
                return " | ".join(detailed_responses)
            else:
                return f"Error: Unable to fetch article summaries. Status code {summary_response.status_code}."
        else:
            return f"Error: Unable to fetch PubMed articles. Status code {response.status_code}."
    except Exception as e:
        return f"Error fetching PubMed articles: {str(e)}"

# Function to fetch drug side effects from the FDA API
import requests

def fetch_drug_side_effects(drug_name: str) -> str:
    """Fetches drug side effects from the FDA API using event data."""
    FDA_API_URL = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&limit=1"
    try:
        response = requests.get(FDA_API_URL)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                side_effects = data['results'][0].get('patient', {}).get('reaction', [])
                if side_effects:
                    # Collect all the side effects descriptions
                    side_effect_descriptions = [reaction.get('reactionmeddrapt', 'No description') for reaction in side_effects]
                    return ", ".join(side_effect_descriptions)
                else:
                    return "No side effects listed for this drug."
            else:
                return "No information available for this drug."
        else:
            return f"Error: {response.status_code} - Unable to fetch data from FDA API"
    except Exception as e:
        return f"Error fetching drug information: {str(e)}"



# Wrap the functions into Langchain tools
pubmed_search_tool = Tool(
    name="fetch_pubmed_articles",
    func=fetch_pubmed_articles,
    description=(
        "This tool fetches research articles from PubMed focused on symptoms, causes of diseases, or treatment options. "
        "Input should be a search term related to a disease or condition followed by the specific query type (e.g., 'fever causes', 'diabetes symptoms', or 'cancer treatment'). "
        "For questions about the causes of diseases, this tool retrieves articles specifically discussing the underlying causes of the disease. "
       "It also provides detailed summaries from PubMed based on the input query, which may cover symptoms, causes, and treatments."
       "If a user asks a complex or partially irrelevant query (e.g., mixing non-medical topics with a question about symptoms or treatments), this tool ensures that the agent only processes the relevant medical parts of the query and retrieves accurate information from PubMed."
    )
)

fda_side_effects_tool = Tool(
    name="fetch_drug_side_effects",
    func=fetch_drug_side_effects,
    description=(
           "This tool fetches detailed side effects of a drug from the FDA API. "
        "Input should be the drug name (e.g., 'paracetamol', 'aspirin'). "
        "If the user asks a complex or irrelevant question but ultimately seeks information about the side effects of a drug (e.g., mixing unrelated topics with the query 'side effects of paracetamol'), this tool will focus only on the relevant drug and fetch the detailed side effects from the FDA API. "
        "Additionally, if the user uses alternative phrases or words similar to 'side effects' (e.g., 'adverse reactions', 'drug impact'), the tool will ensure that the FDA API is called to retrieve comprehensive information about the drug's side effects."
        "ensure that the agent fetch information side effect from articles etc in the FDA "
    )
)

# List of tools for the agent
tools = [pubmed_search_tool, fda_side_effects_tool]

# Define the medical assistant prompt template
medical_prompt_template = """
You are a medical assistant AI. Your primary goal is to provide factual, concise, and respectful responses to healthcare-related questions. You are restricted to responding only to questions related to medical topics such as symptoms, treatments, causes of diseases, drugs, and healthcare advice. You will use PubMed to look up treatment options, causes of diseases, and the FDA API to look up drug side effects.

**Here are some guidelines:**
- Be informative: Provide factual medical information based on available knowledge.
- Be concise: Keep your responses clear and to the point.
- Be respectful: Treat users with courtesy and respect.
- If the question is about symptoms, causes of diseases, or treatment options, call the PubMed API and provide detailed information.
- For questions specifically about the causes of diseases, use the PubMed API to retrieve articles focused on the causes of the disease and provide detailed summaries.
- If the question is about drug side effects, use the FDA API to fetch **detailed side effects** of the drug and provide a comprehensive list of adverse reactions. 


When responding to queries about the causes of diseases (e.g., cancer, fever, diabetes), ensure that you:
- Focus on the underlying causes of the disease.
- Provide article summaries from PubMed that specifically discuss the causes of the disease.

**Please respond to the following query:**
{input}

{agent_scratchpad}
"""


# Define the prompt with the required variables
prompt = PromptTemplate(
    template=medical_prompt_template,
    input_variables=["input", "agent_scratchpad"],  # Include agent_scratchpad
)

# Create the agent and agent executor with correct argument order
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Request model
class QueryRequest(BaseModel):
    question: str

# Endpoint to ask a question to the Langchain agent
@app.post("/ask")
async def ask_question(query: QueryRequest):
    user_input = query.question

    # Invoke the agent with the user input
    try:
        response = agent_executor.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "test123"}},
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Example root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Medical Assistant API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
