# Anime Motion Flow

Self-Supervised Cel-Anime Motion Estimation via Optical Flow and Structural Line-Art Alignment.

## Backend

```powershell
cd backend
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
