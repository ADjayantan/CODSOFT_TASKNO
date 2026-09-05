TRACK_ID=PS08

# Supply Chain Disruption Response Assistant

This project is a solution for the NexusTiq24 Hackathon (PS08). It analyzes unstructured supply chain disruption notices (e.g., supplier emails) and maps them against internal data to assess the impact on stock, shipments, and customer orders.

## Architecture & Rules Compliance
- **Python 3.11 Backend:** A FastAPI server handles backend logic and serves the frontend on port 8000.
- **Gemini API:** Uses `google-generativeai` for LLM structured output generation and `gemini-embedding-001` for the local vector retrieval pipeline.
- **Retrieval Pipeline:** Embeds suppliers and shipments at startup and uses Numpy to perform cosine-similarity searches.
- **Frontend:** HTML/JS/Tailwind served directly by `app.py`. No manual build step required.

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API Key:**
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # Linux/Mac
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. **Start the Application:**
   ```bash
   python app.py
   ```
   The application will start within 90 seconds (indexing the mock DB on startup) and serve on `http://localhost:8000`.

## Generated Data
The application grounds its answers on the JSON data provided in `data/supply_chain.json`. 
It includes:
- **Suppliers:** e.g., Apex Electronics, Global Screens.
- **Shipments:** e.g., SH1001 (500 units of Microchips).
- **Stock:** Current warehouse inventory levels.
- **Orders:** Pending customer orders relying on those parts.

## Demo Video
[Insert Link to 2-3 minute YouTube/Loom Video Here]
