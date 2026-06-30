#!/usr/bin/env bash
set -euo pipefail

echo "Setting up backend with uv..."
cd backend
uv sync
cd ..

echo "Setting up frontend with npm..."
cd frontend
npm install
cd ..

echo "Done. Start the backend with:"
echo "  cd backend && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo "Start the frontend with:"
echo "  cd frontend && npm run dev"
