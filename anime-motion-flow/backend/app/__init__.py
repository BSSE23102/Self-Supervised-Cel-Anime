"""Anime Motion Flow backend package.

The package is split into four layers:

- main.py: FastAPI request/response and job routing.
- processor.py: OpenCV decoding, preprocessing, streaming, and indexing.
- model.py: RAFT/OpenCV optical-flow inference.
- visualization.py: optical-flow rendering for the frontend stream.
"""
