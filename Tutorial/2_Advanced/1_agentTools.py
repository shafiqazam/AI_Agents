from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import ollama

app = FastAPI(title="Local AI Knowledge Base API")

# Initialize persistent local database folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="company_policy"
)

#  Define the request model for incoming queries
class QueryRequest(BaseModel):
    question: str

def fetch_live_policy_allowance(category: str) -> str:
    """Simulates fetching live data from an external API based on the policy category."""
    # Simulate fetching live data from an external API
    # In a real implementation, this would involve making an HTTP request to the API
    # and parsing the response to extract the relevant information.
    # For demonstration purposes, we'll return a hardcoded response based on the category.
    category = category.lower()
    if category == "remote work":
        return "As of today, the company allows up to 3 remote work days per week for eligible employees."
    elif category == "annual leave":
        return "Currently, employees accrue 1.5 days of annual leave per month, totaling 18 days per year."
    elif category == "data protection":
        return "The latest update on data protection policies includes mandatory quarterly training sessions for all staff."
    else:
        return "Sorry, I don't have live data for that policy category at the moment."

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_live_policy_allowance",
            "description": "Get real-time corporate rules or allowance limits for a specific category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "A single word category like 'data protection', 'annual leave', or 'remote work'."
                    }
                },
                "required": ["category"]
            }
        }
    }
]

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        ollama.client.host = "http://host.docker.internal:11434"

        print("Incoming question:", request.question)

        print("Step 1: Commencing ChromaDB Vector Search")
        # Find the corresponding policy related to the search query
        search_results = collection.query(
            query_texts=[request.question],
            n_results=1
        )
        # The answer is in 'documents' key of the search results
        retrieved_context = None
        if search_results and 'documents' in search_results and isinstance(search_results['documents'], list) and len(search_results['documents']) > 0 and search_results['documents'][0]:
            retrieved_context = search_results['documents'][0][0]
            print("Successfully retrieved context:", retrieved_context)

        system_prompt = """You are an internal corporate policy agent. 
                        You have access to tools and vector database context.
                        If a tool is relevant to the user's question, you MUST use it to fetch the data.
                        Answer the user's question based on the provided context ONLY. Keep it short."""
        
        user_prompt = f"Question: {request.question}\n"
        if retrieved_context:
            user_prompt += f"\nContext:\n{retrieved_context}"

        print("Step 2: Sending prompt to local Llama text generator with tool access")

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=AVAILABLE_TOOLS,
            )
        
        message_content = response.get('message', {})
    
        if message_content.get('tool_calls'):
            print("Tool calls detected in response:", message_content['tool_calls'])

            for tool_call in message_content['tool_calls']:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments['category']  # Assuming only one argument for simplicity
                # arguments = tool_call.get('arguments', {})

                print(f"Processing tool call: {function_name} with arguments {arguments}")

                if function_name == "fetch_live_policy_allowance":
                    # Call the actual function to get live data based on the category argument
                    tool_result = fetch_live_policy_allowance(arguments)
                    print(f"Tool result for category '{arguments}': {tool_result}")

                final_response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        # Include the original message content to maintain context, along with the tool call result
                        message_content,
                        {"role": "tool", "name": function_name, "content": f"Tool call result: {tool_result}"}
                    ]
                )
                return {"answer": final_response['message']['content']}
        print("💡 MODEL DECISION: Direct conversational path chosen.")
        return {"answer": message_content.get('content', ''), "source": "Vector Knowledge Base / Static Context"}

    except Exception as e:
        print("🛑 SYSTEM EXCEPTION OCCURRED:", str(e))    

@app.get("/health")
def health_check(): 
    return {"status": "ok", "model": "llama3.2:3b", "vector_store": "ChromaDB"}        