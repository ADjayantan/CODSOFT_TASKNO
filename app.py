import os
import json
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 1. Load Database
db_path = os.path.join(os.path.dirname(__file__), "data", "supply_chain.json")
try:
    with open(db_path, "r") as f:
        db_data = json.load(f)
except FileNotFoundError:
    db_data = {"error": "Database not found"}

# 2. Retrieval Pipeline (In-Memory Vector Store for Suppliers and Shipments)
class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs):
        if not docs: return
        self.documents.extend(docs)
        # Batch embedding
        responses = genai.embed_content(
            model="models/embedding-001",
            content=docs,
            task_type="retrieval_document"
        )
        self.embeddings.extend(responses['embedding'])

    def search(self, query: str, top_k: int = 5):
        if not self.embeddings:
            return []
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(self.embeddings)
        scores = np.dot(doc_vecs, query_vec) / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec))
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]

vector_store = VectorStore()

# 3. Application Startup
@app.on_event("startup")
async def startup_event():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        docs_to_index = []
        for s in db_data.get("suppliers", []):
            docs_to_index.append(f"Supplier: {s['name']} (ID: {s['id']})")
        for sh in db_data.get("shipments", []):
            docs_to_index.append(f"Shipment {sh['shipment_id']}: {sh['quantity']} units of Part {sh['part_id']} from Supplier {sh['supplier_id']}, ETA {sh['eta']}")
        
        try:
            vector_store.add_documents(docs_to_index)
            print("Successfully indexed supply chain database.")
        except Exception as e:
            print(f"Warning: Failed to index: {e}")
    else:
        print("GEMINI_API_KEY not found. Analysis will fail.")

# 4. API Endpoints
class AnalyzeRequest(BaseModel):
    disruption_notice: str

@app.get("/api/database")
async def get_database():
    """Returns the current state of the database for the dashboard."""
    return db_data

@app.get("/api/inbox")
async def get_inbox():
    """Returns simulated incoming supplier emails."""
    return [
        {
            "id": 1,
            "sender": "Apex Electronics",
            "subject": "URGENT: Factory Issue - SH1001 Delayed",
            "body": "Hi team, Apex Electronics here. Unfortunately, we had an issue at the factory and shipment SH1001 will be delayed by 3 weeks.",
            "date": "Today, 09:00 AM"
        },
        {
            "id": 2,
            "sender": "MechParts Inc",
            "subject": "Tracking Update: SH1003",
            "body": "MechParts Inc tracking update: Shipment SH1003 of Aluminum Chassis is delayed by 4 days due to bad weather at the port.",
            "date": "Today, 10:30 AM"
        }
    ]

@app.post("/api/analyze")
async def analyze_disruption(req: AnalyzeRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set. Please set the environment variable.")
        
    try:
        relevant_docs = vector_store.search(req.disruption_notice, top_k=3)
        retrieved_context = "\n".join(relevant_docs)
    except Exception as e:
        retrieved_context = "Error during retrieval."

    db_string = json.dumps(db_data, indent=2)

    system_instruction = f"""
    You are a supply chain disruption response assistant.
    
    RETRIEVED ENTITIES:
    {retrieved_context}
    
    FULL SUPPLY CHAIN DATABASE:
    {db_string}

    The user will provide an unstructured disruption notice.
    Your job is to:
    1. Identify which supplier, shipment, or part is affected.
    2. Trace the impact to current warehouse stock and in-transit shipments.
    3. Calculate the impact on pending customer orders.
    4. Provide an impact assessment and an action plan with trade-offs.
    5. MUST ground every claim to the database. If notice maps to nothing, state 'No Impact'.
    
    Respond in strict JSON matching this structure:
    {{
      "has_impact": boolean,
      "summary": "string",
      "affected_orders": [
        {{
          "order_id": "string",
          "customer": "string",
          "urgency": "High/Medium/Low",
          "impact_details": "string",
          "options": [
             {{
               "action": "string",
               "trade_offs": "string"
             }}
          ]
        }}
      ],
      "recommended_course": "string"
    }}
    """

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"}
    )
    
    try:
        response = model.generate_content(req.disruption_notice)
        return json.loads(response.text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=repr(e))

# 5. Frontend Serving
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
