# Anime Motion Flow - Technical Project Explanation

## 1. Project Goal

Anime Motion Flow is a full-stack computer vision research prototype for analyzing motion in cel/anime-style MP4 clips.

The research theme is:

Self-Supervised Cel-Anime Motion Estimation via Optical Flow and Structural Line-Art Alignment

In practical terms, the system lets a user upload an anime action clip, processes consecutive video frames, estimates pixel-level motion, renders a clean motion-vector overlay, and indexes the detected motion so users can search for action segments by direction and velocity.

The project is intentionally split into two decoupled applications:

```text
anime-motion-flow/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI routes and streaming API
│   │   ├── model.py         RAFT/OpenCV optical-flow inference
│   │   ├── processor.py     Video decoding, frame processing, metadata extraction
│   │   └── visualization.py Clean motion-vector rendering
│   └── pyproject.toml       Python dependencies managed by uv
├── frontend/
│   ├── src/App.tsx          React TypeScript UI
│   └── vite.config.ts       Vite configuration
└── explanation.me           This technical explanation
```

## 2. High-Level System Flow

```text
User
  |
  | selects or drops MP4
  v
React + TypeScript Frontend
  |
  | POST /api/process-video
  v
FastAPI Backend
  |
  | stores upload in temporary MP4 file
  | creates job_id
  v
Video Stream Endpoint
  |
  | GET /api/process-video/{job_id}/stream
  v
OpenCV VideoCapture
  |
  | reads frame_t and frame_t+1
  v
Motion Reference Preprocessing
  |
  | bilateral smoothing -> grayscale -> Gaussian blur
  v
Optical Flow Inference
  |
  | RAFT if enabled, otherwise OpenCV Farneback fallback
  v
Flow Tensor F(x, y) = [u(x, y), v(x, y)]
  |
  +--> Clean vector visualization streamed as multipart JPEG frames
  |
  +--> Motion metadata extraction saved to motion_index.json
                 |
                 v
       GET /api/search-actions
                 |
                 v
       Direction/velocity segment search with representative thumbnails
```

## 3. Backend Architecture

The backend is a FastAPI service.

Important files:

```text
backend/app/main.py
```

Owns the HTTP API, CORS, job registry, streaming endpoint, preview endpoint, frame thumbnail endpoint, and action search endpoint.

```text
backend/app/model.py
```

Owns optical-flow inference. It can use RAFT through TorchVision if enabled with an environment variable, but defaults to OpenCV optical flow for interactive stability on local machines.

```text
backend/app/processor.py
```

Owns MP4 saving, frame decoding, preprocessing, metadata extraction, JPEG encoding, and multipart stream generation.

```text
backend/app/visualization.py
```

Owns the clean vector rendering layer. The live stream currently preserves the original frame and overlays sparse, high-confidence green vectors. The dense HSV heatmap is retained as an optional diagnostic mode but is disabled by default.

## 4. Backend API

### GET /health

Returns:

```json
{"status": "ok"}
```

The frontend calls this before upload to confirm the backend is running.

### POST /api/process-video

Accepts an MP4 upload:

```text
multipart/form-data
file: video/mp4
```

Behavior:

1. Validates that the upload looks like an MP4.
2. Creates a random `job_id`.
3. Saves the video to a temporary local file.
4. Stores the job in an in-memory dictionary.
5. Returns the stream URL.

Example response:

```json
{
  "job_id": "abc123...",
  "filename": "anime_action_clip.mp4",
  "stream_url": "/api/process-video/abc123.../stream",
  "status": "queued"
}
```

### GET /api/process-video/{job_id}/stream

Streams processed JPEG frames using:

```text
multipart/x-mixed-replace
```

This is why the frontend can display the processed output in a normal `<img />` tag. The browser keeps replacing the image as new JPEG parts arrive.

### GET /api/process-video/{job_id}/preview

Returns the first decoded frame as a JPEG.

### GET /api/process-video/{job_id}/frame/{frame_index}

Returns a specific video frame as a JPEG. The action-search UI uses this to show representative thumbnails for detected motion segments.

### GET /api/search-actions?direction=right&min_velocity=5&job_id=...

Reads `motion_index.json`, filters motion metadata, groups consecutive matching frames into scene segments, and returns segment timestamps plus thumbnail URLs.

Example segment:

```json
{
  "direction": "right",
  "start_frame": 19,
  "end_frame": 25,
  "representative_frame": 22,
  "thumbnail_url": "/api/process-video/{job_id}/frame/22",
  "start_timestamp": 0.63,
  "end_timestamp": 0.83,
  "frame_count": 7,
  "mean_velocity": 12.4,
  "peak_velocity": 18.9
}
```

## 5. Frontend Architecture

The frontend is a Vite React + TypeScript application.

Main file:

```text
frontend/src/App.tsx
```

It provides:

- MP4 dropzone and file picker.
- Upload progress tracking.
- Backend health check.
- Side-by-side original video and processed motion stream.
- Action search controls for direction and minimum velocity.
- Search results with visual representative frames.

Important frontend state:

```ts
file: File | null
jobId: string | null
originalUrl: string | null
streamUrl: string | null
progress: number
status: 'idle' | 'ready' | 'uploading' | 'streaming' | 'error'
searchDirection: 'left' | 'right' | 'up' | 'down'
minVelocity: number
searchResults: SearchActionsResponse | null
```

Frontend API sequence:

```text
1. User selects MP4
2. Frontend creates local object URL for original video preview
3. Frontend calls GET /health
4. Frontend posts file to POST /api/process-video
5. Backend returns job_id and stream_url
6. Frontend sets <img src={stream_url}>
7. Browser receives multipart JPEG stream
8. User searches indexed actions with GET /api/search-actions
```

## 6. Video Processing Pipeline

The backend processes the uploaded video frame by frame.

Let a video be a sequence:

```text
I_1, I_2, I_3, ..., I_T
```

Each `I_t` is an RGB/BGR image with shape:

```text
H x W x 3
```

The pipeline compares consecutive frames:

```text
(I_1, I_2), (I_2, I_3), ..., (I_{T-1}, I_T)
```

For every pair, the system estimates optical flow:

```text
F_t(x, y) = [u_t(x, y), v_t(x, y)]
```

Where:

- `u_t(x, y)` is horizontal displacement at pixel `(x, y)`.
- `v_t(x, y)` is vertical displacement at pixel `(x, y)`.
- Positive `u` means motion to the right.
- Positive `v` means motion downward in image coordinates.

## 7. Motion Reference Preprocessing

The current implementation estimates motion from smoothed luminance frames, not dense edge maps. This was changed because edge-only maps can create unstable vectors in anime footage: small line-art flicker, compression artifacts, and outlines can produce false motion.

Current preprocessing:

```text
original BGR frame
  -> bilateral filter
  -> grayscale conversion
  -> Gaussian blur
  -> 2D motion reference frame
```

### 7.1 Bilateral Filter

The bilateral filter smooths noise while preserving edges.

For pixel `p`, the filtered value is:

```text
I_filtered(p) =
    (1 / W_p) * sum_{q in neighborhood} I(q)
        * exp(-||p - q||^2 / (2 sigma_s^2))
        * exp(-||I(p) - I(q)||^2 / (2 sigma_r^2))
```

Where:

- `sigma_s` controls spatial distance weighting.
- `sigma_r` controls color/intensity difference weighting.
- `W_p` is a normalization factor.

Why it matters for anime:

- Anime has large flat color regions and sharp outlines.
- Normal blur can wash out lines.
- Bilateral filtering reduces compression noise while keeping meaningful boundaries.

### 7.2 Grayscale Conversion

The BGR frame is converted to a single-channel luminance image:

```text
Y = 0.114 B + 0.587 G + 0.299 R
```

This gives optical flow a stable intensity field without depending on color changes.

### 7.3 Gaussian Blur

A small Gaussian blur suppresses local noise:

```text
G(x, y) = (1 / (2 pi sigma^2)) * exp(-(x^2 + y^2) / (2 sigma^2))
```

The processed motion reference is:

```text
R_t = GaussianBlur(Grayscale(BilateralFilter(I_t)))
```

Optical flow is then estimated between:

```text
R_t and R_{t+1}
```

## 8. Optical Flow Math

Optical flow estimates how pixels move between two consecutive frames.

The classical assumption is brightness constancy:

```text
I(x, y, t) = I(x + u, y + v, t + 1)
```

This means the same visual point keeps roughly the same brightness while moving.

For small motion, using a first-order Taylor expansion:

```text
I(x + u, y + v, t + 1)
  approx I(x, y, t) + I_x u + I_y v + I_t
```

Substituting into brightness constancy gives:

```text
I_x u + I_y v + I_t = 0
```

Where:

- `I_x` is horizontal image gradient.
- `I_y` is vertical image gradient.
- `I_t` is temporal difference between frames.
- `u` and `v` are the unknown pixel displacement components.

This single equation has two unknowns, so optical-flow algorithms add extra constraints such as local smoothness, multi-scale estimation, feature correlation, or learned priors.

## 9. RAFT Inference Path

The backend supports TorchVision RAFT:

```python
raft_large(weights=Raft_Large_Weights.DEFAULT)
```

RAFT stands for Recurrent All-Pairs Field Transforms.

Conceptually, RAFT does:

```text
Frame A, Frame B
  -> feature encoder
  -> all-pairs correlation volume
  -> recurrent update block
  -> dense flow field
```

### 9.1 Feature Extraction

RAFT encodes both frames into learned feature maps:

```text
f_1 = encoder(I_1)
f_2 = encoder(I_2)
```

Each pixel has a feature vector.

### 9.2 All-Pairs Correlation

RAFT builds a correlation volume comparing every feature in frame 1 with every feature in frame 2:

```text
C(i, j) = <f_1(i), f_2(j)>
```

Where:

- `i` indexes a pixel/feature position in frame 1.
- `j` indexes a pixel/feature position in frame 2.
- `<., .>` is a dot product.

This lets RAFT reason about long-range motion, which is important for action scenes like sword slashes, fast punches, or camera pans.

### 9.3 Recurrent Flow Refinement

RAFT starts with an initial flow estimate and refines it repeatedly:

```text
F^{k+1} = F^k + DeltaF^k
```

The update block uses correlation lookups and context features to improve the flow field.

In this project, RAFT is optional because loading CUDA Torch wheels can be heavy on Windows. The current default is:

```text
ANIME_FLOW_RAFT_MODE=opencv
```

To enable RAFT:

```powershell
$env:ANIME_FLOW_RAFT_MODE="cuda"
```

or:

```powershell
$env:ANIME_FLOW_RAFT_MODE="cpu"
```

## 10. OpenCV Farneback Fallback

When RAFT is disabled or unavailable, the backend uses OpenCV Farneback dense optical flow.

Farneback approximates local neighborhoods using quadratic polynomials:

```text
I_1(x) approx x^T A_1 x + b_1^T x + c_1
I_2(x) approx x^T A_2 x + b_2^T x + c_2
```

If the neighborhood moves by displacement `d`, then:

```text
I_2(x) approx I_1(x - d)
```

The algorithm estimates `d = [u, v]` by comparing polynomial coefficients across frames.

Implementation details:

- The frame is resized to half resolution for speed.
- Farneback computes flow at the smaller resolution.
- The flow is resized back to original resolution.
- The `u` and `v` channels are scaled back to preserve pixel displacement units.

This gives interactive performance for local demos while preserving the same output tensor format as RAFT:

```text
flow.shape = H x W x 2
```

## 11. Flow Tensor Representation

The final flow tensor stores:

```text
flow[y, x, 0] = u(x, y)
flow[y, x, 1] = v(x, y)
```

For each pixel:

```text
F(x, y) = [u(x, y), v(x, y)]
```

Magnitude:

```text
M(x, y) = sqrt(u(x, y)^2 + v(x, y)^2)
```

Angle:

```text
theta(x, y) = atan2(v(x, y), u(x, y))
```

In image coordinates:

- `theta = 0 degrees` means right.
- `theta = 90 degrees` means down.
- `theta = 180 degrees` means left.
- `theta = 270 degrees` means up.

## 12. Clean Motion Vector Rendering

The current renderer is designed to avoid messy full-frame arrow fields.

The default output is:

```text
original frame + sparse high-confidence green vectors
```

The optional HSV heatmap is disabled by default because it can obscure the anime frame and confuse viewers. It is useful only as a research diagnostic when you want dense direction/magnitude information.

### 12.1 Why Dense Heatmaps Can Be Confusing

An HSV optical flow map encodes:

- Direction as hue.
- Speed as brightness/value.

That is mathematically valid, but visually it can look like arbitrary color blobs if the audience does not already understand optical-flow color coding.

For presentation and qualitative inspection, sparse vectors are easier:

- Arrow direction shows motion direction.
- Arrow length shows relative velocity.
- The original scene stays visible.
- Only high-motion regions are emphasized.

### 12.2 Adaptive Threshold

The renderer computes motion magnitude:

```text
M = sqrt(u^2 + v^2)
```

Then it selects only strong motion:

```text
tau = max(percentile(M[M > 0], 98), 6.0)
```

Where:

- `percentile(..., 98)` keeps only the fastest moving pixels in the current frame.
- `6.0` is a minimum pixel/frame velocity threshold.

This avoids drawing arrows on static backgrounds.

### 12.3 Sparse Sampling Grid

Instead of drawing an arrow at every pixel, the renderer samples a grid:

```text
step = 40 pixels
```

Candidate arrow centers are:

```text
(x, y) = (20, 20), (60, 20), (100, 20), ...
```

This makes the visualization readable.

### 12.4 Local Vector Averaging

At each grid point, the renderer examines a local window `W`.

Active pixels are:

```text
A = {p in W | M(p) >= tau}
```

If the active area is too small, no arrow is drawn.

For valid windows, the vector is magnitude-weighted:

```text
u_bar = sum_{p in A} M(p) * u(p) / sum_{p in A} M(p)
v_bar = sum_{p in A} M(p) * v(p) / sum_{p in A} M(p)
```

The arrow starts near the weighted center of active motion:

```text
x_bar = sum_{p in A} M(p) * x_p / sum_{p in A} M(p)
y_bar = sum_{p in A} M(p) * y_p / sum_{p in A} M(p)
```

The arrow endpoint is:

```text
end = start + scale * [u_bar, v_bar]
```

The renderer caps the maximum visual arrow length so very large displacements do not dominate the frame.

### 12.5 Alpha Compositing

The arrow layer starts as a black image:

```text
arrow_layer = zeros_like(frame)
```

The final output is:

```text
output = addWeighted(frame, 1.0, arrow_layer, 0.95, 0)
```

Because the arrow layer is black except where arrows are drawn, the original frame stays visible and only green arrows are added.

## 13. Optional HSV Diagnostic Layer

The visualization code still supports an HSV layer with:

```python
show_heatmap=True
```

The HSV mapping is:

```text
Hue   = theta / 2      because OpenCV hue range is [0, 179]
Sat   = 180
Value = robust_normalize(M)
```

Robust normalization:

```text
low  = percentile(M[M > 0], activation_percentile)
high = percentile(M[M > 0], saturation_percentile)
V    = clip((M - low) / (high - low), 0, 1)^1.65 * 220
```

Purpose:

- Good for papers/debugging when you want dense per-pixel motion.
- Bad as the default UI output because it hides the source animation.

Current product decision:

```text
Heatmap off by default.
Sparse vector overlay on by default.
```

## 14. Motion Metadata Extraction

For every frame pair, the system extracts a compact action descriptor.

Input:

```text
flow_tensor = H x W x 2
frame_index
fps
```

Magnitude and angle:

```text
M(x, y) = sqrt(u^2 + v^2)
theta(x, y) = atan2(v, u)
```

Moving pixels:

```text
moving_mask = M > 2.0
```

Average velocity:

```text
avg_velocity = mean(M[moving_mask])
```

If no pixels pass the threshold:

```text
avg_velocity = 0
direction = "static"
```

Timestamp:

```text
timestamp = frame_index / fps
```

Output record:

```json
{
  "frame": 35,
  "timestamp": 1.1667,
  "direction": "right",
  "avg_velocity": 7.52
}
```

## 15. Direction Classification

The system bins optical-flow angles into four dominant directions.

OpenCV angle convention:

```text
0 degrees     right
90 degrees    down
180 degrees   left
270 degrees   up
```

Bins:

```text
right: theta < 45 or theta >= 315
down:  45 <= theta < 135
left:  135 <= theta < 225
up:    225 <= theta < 315
```

The dominant direction is the bin with the most moving pixels.

This is simple and presentation-friendly. It turns a dense optical-flow field into a searchable semantic tag.

## 16. Motion Index and Action Search

At stream completion, the backend writes:

```text
backend/motion_index.json
```

This file stores one metadata record per processed frame pair.

Search query:

```text
direction = "right"
min_velocity = 5.0
```

Filtering:

```text
matches = [
  item
  for item in motion_index
  if item.direction == direction
  and item.avg_velocity >= min_velocity
]
```

Consecutive frame grouping:

```text
if current_frame == previous_frame + 1:
    same segment
else:
    new segment
```

Segment statistics:

```text
start_timestamp = first_match.timestamp
end_timestamp   = last_match.timestamp
mean_velocity   = mean(segment velocities)
peak_velocity   = max(segment velocities)
```

Representative frame:

```text
representative_frame = middle frame of segment
```

The frontend uses this representative frame to show a visual result card.

## 17. Why This Is Useful for Anime Motion Research

Anime motion is different from natural video:

- Large flat color regions.
- Strong line art.
- Abrupt cuts and stylized smears.
- Limited animation frames.
- Fast effects such as slashes, impact frames, explosions, and speed lines.

This system is useful because it produces:

- Dense flow internally for quantitative analysis.
- Sparse vectors externally for readable diagnostics.
- Motion metadata for searching action fragments.
- Frame thumbnails for qualitative review.

The action-search feature is especially helpful for research datasets. Instead of manually scrubbing video, a researcher can ask:

```text
Find rightward fast motion above 8 pixels/frame.
Find upward motion during jumps.
Find leftward action cuts.
```

## 18. Current Implementation Choices

### Default Flow Engine

The default engine is OpenCV Farneback:

```text
ANIME_FLOW_RAFT_MODE=opencv
```

Reason:

- Faster local demo.
- Avoids heavy Torch/CUDA memory issues on Windows.
- Still produces a dense flow field compatible with the rest of the pipeline.

### Optional RAFT

RAFT remains available if the environment can load Torch/TorchVision cleanly:

```text
ANIME_FLOW_RAFT_MODE=cuda
```

or:

```text
ANIME_FLOW_RAFT_MODE=cpu
```

### Clean Visualization

The project currently prioritizes presentation clarity:

```text
original frame + sparse high-motion arrows
```

The heatmap exists only as an optional diagnostic.

## 19. Important Limitations

This is a research prototype, not yet a distributed production system.

Current limitations:

- `motion_index.json` is a single local file.
- Jobs are stored in process memory, so they reset when the backend restarts.
- Uploaded videos are temporary files and should be cleaned up for long-running deployments.
- Search is rule-based, not FAISS/vector-database based in the current code.
- RAFT is optional and may fail on systems without enough GPU/CPU memory.
- Multipart JPEG streaming is simple and browser-friendly, but not as efficient as HLS/WebRTC for large deployments.

Recommended production upgrades:

- Store jobs in Redis or a database.
- Store motion indexes per job ID.
- Add background workers with Celery/RQ/Arq.
- Store uploaded videos and outputs in object storage.
- Add FAISS embeddings for similarity search over motion descriptors.
- Add authentication and rate limiting.
- Add batch processing and dataset export.
- Add structured experiment tracking with MLflow or Weights & Biases.

## 20. FAISS Extension Concept

The current search uses exact filtering:

```text
direction == query_direction
avg_velocity >= threshold
```

A FAISS-based version would convert each frame or segment into a vector:

```text
z_t = [
  mean_u,
  mean_v,
  mean_magnitude,
  peak_magnitude,
  direction_one_hot_right,
  direction_one_hot_left,
  direction_one_hot_up,
  direction_one_hot_down
]
```

For richer segment embeddings:

```text
z_segment = [
  mean velocity,
  peak velocity,
  duration,
  histogram of flow angles,
  histogram of magnitudes,
  spatial centroid of motion,
  motion area ratio
]
```

FAISS would index:

```text
IndexFlatL2 or IndexHNSWFlat
```

Nearest-neighbor query:

```text
argmin_i ||z_query - z_i||_2
```

This would support semantic searches such as:

```text
"Find clips similar to this slash motion."
"Find fast diagonal attacks."
"Find scenes with camera pan plus character motion."
```

## 21. Presentation Script

A strong way to explain the project:

1. "This project analyzes anime motion by estimating dense optical flow between consecutive frames."
2. "The backend accepts an MP4, decodes it with OpenCV, and creates smoothed luminance frame pairs."
3. "Each pair is passed through RAFT if enabled, otherwise through OpenCV Farneback, producing a two-channel flow tensor."
4. "For every pixel, the flow tensor stores horizontal and vertical displacement: u and v."
5. "From u and v we compute magnitude and direction using sqrt(u^2 + v^2) and atan2(v, u)."
6. "Instead of visualizing every pixel, we use an adaptive threshold and sparse local averaging to draw only meaningful high-motion vectors."
7. "At the same time, we compress the dense flow into searchable metadata: timestamp, dominant direction, and average velocity."
8. "The frontend streams the processed output live and lets users search motion segments visually with thumbnails."
9. "The result is both a qualitative diagnostic tool and the beginning of a motion-indexing system for anime datasets."

## 22. Key Equations to Put on Slides

Brightness constancy:

```text
I(x, y, t) = I(x + u, y + v, t + 1)
```

Optical flow constraint:

```text
I_x u + I_y v + I_t = 0
```

Flow vector:

```text
F(x, y) = [u(x, y), v(x, y)]
```

Magnitude:

```text
M(x, y) = sqrt(u(x, y)^2 + v(x, y)^2)
```

Angle:

```text
theta(x, y) = atan2(v(x, y), u(x, y))
```

Adaptive threshold:

```text
tau = max(P_98(M), 6.0)
```

Weighted local vector:

```text
u_bar = sum M_i u_i / sum M_i
v_bar = sum M_i v_i / sum M_i
```

Timestamp:

```text
time = frame_index / fps
```

## 23. One-Slide Architecture Summary

```text
React UI
  - MP4 upload
  - original preview
  - live processed stream
  - action search dashboard

FastAPI API
  - upload endpoint
  - multipart stream endpoint
  - frame thumbnail endpoint
  - search endpoint

Computer Vision Core
  - OpenCV decoding
  - bilateral + grayscale + blur preprocessing
  - RAFT or Farneback optical flow
  - sparse vector rendering
  - motion metadata indexing
```

## 24. How to Defend the Heatmap Decision

If asked "why not show the heatmap?", answer:

The HSV heatmap is scientifically useful because it encodes dense direction and speed at every pixel. However, for anime footage it can obscure scene content and make qualitative interpretation harder. The project therefore keeps HSV rendering as an optional diagnostic but defaults to sparse high-confidence vectors, which are easier to explain in a presentation and better for visual inspection.

## 25. What Makes This Project Advanced

- It bridges ML/CV backend work with a usable frontend.
- It uses frame-pair optical flow to infer motion without manual labels.
- It supports RAFT, a modern learned optical-flow architecture.
- It includes a practical fallback for local development.
- It converts dense tensor output into human-readable visual diagnostics.
- It creates a searchable temporal motion index.
- It shows how research outputs can become interactive tools.

