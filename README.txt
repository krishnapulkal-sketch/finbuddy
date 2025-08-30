FinBuddy (Single-Page App + FastAPI backend)

HOW TO RUN
1) Create venv (recommended) and install deps:
   python -m venv .venv
   .venv\Scripts\activate   (Windows)  OR  source .venv/bin/activate (macOS/Linux)
   pip install -r requirements.txt

2) (Optional) copy .env.example to .env and fill API keys. Without keys it will still work
   with dummy responses.

3) Start the server from this folder:
   uvicorn app:app --reload --port 8000

4) Open the app:
   http://127.0.0.1:8000/

NOTES
- All tabs are within one page (index.html). The backend serves SPA + APIs.
- If you change the frontend, files are in frontend/ and auto-served via /assets.
- Default user id in the UI is "demo". Change on the Settings tab to separate data.
