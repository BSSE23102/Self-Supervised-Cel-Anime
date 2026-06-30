$ErrorActionPreference = "Stop"

Write-Host "Setting up backend with uv..."
Push-Location backend
uv sync
Pop-Location

Write-Host "Setting up frontend with npm..."
Push-Location frontend
npm install
Pop-Location

Write-Host "Done. Start the backend with:"
Write-Host "  cd backend; uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000"
Write-Host "Start the frontend with:"
Write-Host "  cd frontend; npm run dev"
