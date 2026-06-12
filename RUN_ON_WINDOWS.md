# Run Locally on Windows 11 (Core i5, CPU-only, OCR removed)

OCR (EasyOCR / Ollama / Gemini / OpenAI header OCR) has been removed from
`python-multiline/main.py` so the project runs on a normal laptop without a GPU
or API keys. The API endpoints and response shapes are unchanged, so the
frontend works as-is. OCR-related fields simply come back empty with
`"engine": "disabled"`. See the notes at the bottom for re-integrating OCR later.

## Prerequisites (install once)

1. **Python 3.11** (or 3.10/3.12) - https://www.python.org/downloads/ (tick "Add to PATH")
2. **Node.js 20 LTS** - https://nodejs.org/
3. **MongoDB Community Server** - https://www.mongodb.com/try/download/community
   (only needed for login/signup/projects; the main curve-extraction dashboard talks
   directly to the Python service)

## 1. Start the Python service (Terminal 1)

```powershell
cd python-multiline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Tip: to install CPU-only PyTorch (much smaller download), run this BEFORE
`pip install -r requirements.txt`:

```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

## 2. Start the Node backend (Terminal 2)

```powershell
cd js-backend
npm install
npm start
```

## 3. Start the frontend (Terminal 3)

```powershell
cd frontend
npm install
npm run dev
```

## URLs / IPs to use

| Service              | Address (open in browser / used by app)   |
|----------------------|--------------------------------------------|
| Frontend (main app)  | http://localhost:5173                      |
| Python FastAPI       | http://127.0.0.1:8000                      |
| Python API docs      | http://127.0.0.1:8000/docs                 |
| Node/Express backend | http://127.0.0.1:5000                      |
| MongoDB              | mongodb://localhost:27017/graphocr         |

Open **http://localhost:5173** in your browser to use the app.
Vite runs with `--host`, so from another device on the same Wi-Fi you can also
open `http://<your-PC-LAN-IP>:5173` (find it with `ipconfig` -> IPv4 Address).

The `.env` files are already created and point everything to these local
addresses:

- `frontend/.env` -> Python at 127.0.0.1:8000, Node at 127.0.0.1:5000
- `js-backend/.env` -> port 5000, local MongoDB
- `python-multiline/.env` -> no API keys needed anymore

## Quick API test (without the frontend)

```powershell
curl -X POST http://127.0.0.1:8000/segment-and-graph -F "file=@path\to\well_log.tif" -F "threshold=127" -F "total_graphs=1"
```

## Notes about removed OCR

What was removed/stubbed in `python-multiline/main.py`:

- `import easyocr`, `header_ocr_engine` (Ollama), Google Gemini and OpenAI clients
- `extract_las_header()` -> now returns empty header with `"engine": "disabled"`
- `extract_depth_ticks_ocr()` -> now returns `[]` (use the manual depth range
  inputs in the UI instead)
- axis-label OCR inside `match_graph_curves_to_values()` -> empty label list

What still works: layout detection, TIFF preprocessing, curve tracking/extraction,
point editing, and LAS export. Model weight files (`models/extract_header_model.pt`,
`best.pt`) were not included in the zip; the code falls back to density-based
layout detection when they are missing.

To re-integrate OCR later: uncomment the OCR packages in
`python-multiline/requirements.txt`, reinstall, and restore the original
function bodies (each stub has a comment marking it).
