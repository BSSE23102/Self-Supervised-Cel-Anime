import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import {
  Activity,
  CheckCircle2,
  CircleAlert,
  Clock3,
  Eraser,
  FileVideo,
  Gauge,
  Image,
  Loader2,
  Play,
  Search,
  UploadCloud,
} from 'lucide-react'

type ProcessVideoResponse = {
  job_id: string
  filename: string
  stream_url: string
  status: string
}

type RunState = 'idle' | 'ready' | 'uploading' | 'streaming' | 'error'
type MotionDirection = 'left' | 'right' | 'up' | 'down'

type ActionSegment = {
  direction: string
  start_frame: number
  end_frame: number
  representative_frame: number
  thumbnail_url: string
  start_timestamp: number
  end_timestamp: number
  frame_count: number
  mean_velocity: number
  peak_velocity: number
}

type SearchActionsResponse = {
  query: {
    direction: string
    min_velocity: number
    job_id: string | null
  }
  match_count: number
  segments: ActionSegment[]
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const statusCopy: Record<RunState, string> = {
  idle: 'Awaiting clip',
  ready: 'Ready',
  uploading: 'Uploading',
  streaming: 'Indexed stream',
  error: 'Action needed',
}

const directions: MotionDirection[] = ['right', 'left', 'up', 'down']

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatSeconds(seconds: number) {
  return `${seconds.toFixed(2)}s`
}

function getErrorMessage(err: unknown) {
  if (axios.isAxiosError(err)) {
    if (err.code === 'ECONNABORTED') {
      return 'The backend did not respond. Check that FastAPI is running on port 8000.'
    }
    const detail = err.response?.data?.detail
    return typeof detail === 'string' ? detail : err.message
  }
  return 'The request could not be completed.'
}

function buildMediaUrl(path: string) {
  return path.startsWith('http') ? path : `${API_URL}${path}`
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [originalUrl, setOriginalUrl] = useState<string | null>(null)
  const [streamUrl, setStreamUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<RunState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)
  const [searchDirection, setSearchDirection] = useState<MotionDirection>('right')
  const [minVelocity, setMinVelocity] = useState(5)
  const [isSearching, setIsSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<SearchActionsResponse | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    return () => {
      if (originalUrl) URL.revokeObjectURL(originalUrl)
    }
  }, [originalUrl])

  const acceptFile = (candidate?: File) => {
    if (!candidate) return

    const isMp4 =
      candidate.type === 'video/mp4' || candidate.name.toLowerCase().endsWith('.mp4')

    if (!isMp4) {
      setStatus('error')
      setError('Select an MP4 clip before processing.')
      return
    }

    if (originalUrl) URL.revokeObjectURL(originalUrl)

    setFile(candidate)
    setJobId(null)
    setOriginalUrl(URL.createObjectURL(candidate))
    setStreamUrl(null)
    setStreamError(null)
    setProgress(0)
    setError(null)
    setSearchError(null)
    setSearchResults(null)
    setStatus('ready')
  }

  const clearProject = () => {
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    setFile(null)
    setJobId(null)
    setOriginalUrl(null)
    setStreamUrl(null)
    setStreamError(null)
    setIsDragging(false)
    setIsUploading(false)
    setProgress(0)
    setStatus('idle')
    setError(null)
    setSearchError(null)
    setSearchResults(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const processClip = async () => {
    if (!file) {
      setStatus('error')
      setError('Choose an anime action clip first.')
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    setIsUploading(true)
    setProgress(0)
    setStreamUrl(null)
    setStreamError(null)
    setError(null)
    setSearchError(null)
    setSearchResults(null)
    setStatus('uploading')

    try {
      await axios.get(`${API_URL}/health`, { timeout: 5000 })

      const response = await axios.post<ProcessVideoResponse>(
        `${API_URL}/api/process-video`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 45000,
          onUploadProgress: (event) => {
            if (!event.total) return
            const nextProgress = Math.round((event.loaded / event.total) * 100)
            setProgress(Math.min(nextProgress, 99))
          },
        },
      )

      const nextStreamUrl = buildMediaUrl(response.data.stream_url)
      const cacheKey = Date.now()
      const liveStreamUrl = `${nextStreamUrl}${nextStreamUrl.includes('?') ? '&' : '?'}t=${cacheKey}`

      setJobId(response.data.job_id)
      setStreamUrl(liveStreamUrl)
      setStreamError(null)
      setProgress(100)
      setStatus('streaming')
    } catch (err) {
      setStatus('error')
      setError(getErrorMessage(err))
      setProgress(0)
    } finally {
      setIsUploading(false)
    }
  }

  const searchActions = async () => {
    setIsSearching(true)
    setSearchError(null)

    try {
      const response = await axios.get<SearchActionsResponse>(
        `${API_URL}/api/search-actions`,
        {
          params: {
            direction: searchDirection,
            min_velocity: minVelocity,
            job_id: jobId,
          },
          timeout: 10000,
        },
      )
      setSearchResults(response.data)
    } catch (err) {
      setSearchResults(null)
      setSearchError(getErrorMessage(err))
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <main className="min-h-screen bg-[#eef2f0] text-[#17211d]">
      <div className="fixed inset-0 -z-10 bg-[linear-gradient(135deg,rgba(8,86,82,0.09)_0%,rgba(238,242,240,0)_36%),radial-gradient(circle_at_92%_14%,rgba(201,133,37,0.16),rgba(238,242,240,0)_25%)]" />

      <section className="mx-auto flex min-h-screen w-full max-w-[1600px] flex-col px-4 py-4 md:px-6">
        <header className="grid gap-4 border-b border-[#c5d0cb] pb-4 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <div className="mb-3 flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-lg bg-[#0a5c5a] text-white shadow-sm">
                <Activity className="h-5 w-5" aria-hidden="true" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#68746f]">
                  Cel-animation motion lab
                </p>
                <h1 className="text-2xl font-semibold tracking-normal text-[#121a17] md:text-3xl">
                  Anime Motion Flow
                </h1>
              </div>
            </div>
            <p className="max-w-3xl text-sm leading-6 text-[#53605b]">
              Compare source video, rendered motion fields, and searchable action segments.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs font-medium">
            <span className="rounded-lg border border-[#c7d0cc] bg-white/80 px-3 py-2 text-[#53605b]">
              API {API_URL.replace(/^https?:\/\//, '')}
            </span>
            <span className="inline-flex items-center gap-2 rounded-lg border border-[#b9cbc6] bg-[#e1efeb] px-3 py-2 text-[#0a5c5a]">
              {status === 'error' ? (
                <CircleAlert className="h-4 w-4" aria-hidden="true" />
              ) : status === 'streaming' ? (
                <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
              ) : (
                <Activity className="h-4 w-4" aria-hidden="true" />
              )}
              {statusCopy[status]}
            </span>
          </div>
        </header>

        <div className="grid flex-1 gap-4 py-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="flex flex-col gap-4">
            <section
              className={[
                'rounded-lg border bg-white p-4 shadow-sm transition',
                isDragging ? 'border-[#0a5c5a] ring-4 ring-[#0a5c5a]/10' : 'border-[#cfd8d4]',
              ].join(' ')}
              onDragOver={(event) => {
                event.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setIsDragging(false)
                acceptFile(event.dataTransfer.files[0])
              }}
            >
              <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold text-[#17211d]">Source clip</h2>
                  <p className="mt-1 text-xs leading-5 text-[#65716d]">
                    MP4 input for frame-pair motion estimation.
                  </p>
                </div>
                <UploadCloud className="h-5 w-5 text-[#0a5c5a]" aria-hidden="true" />
              </div>

              <button
                type="button"
                className="group flex min-h-36 w-full flex-col items-center justify-center rounded-lg border border-dashed border-[#b6c4bf] bg-[#f7faf8] px-4 py-5 text-center transition hover:border-[#0a5c5a] hover:bg-[#edf7f3]"
                onClick={() => fileInputRef.current?.click()}
              >
                <FileVideo className="mb-3 h-8 w-8 text-[#0a5c5a]" aria-hidden="true" />
                <span className="max-w-full break-words text-sm font-semibold text-[#17211d]">
                  {file ? file.name : 'anime_action_clip.mp4'}
                </span>
                <span className="mt-1 text-xs text-[#65716d]">
                  {file ? formatBytes(file.size) : 'Drop file here or browse'}
                </span>
              </button>

              <input
                ref={fileInputRef}
                className="sr-only"
                type="file"
                accept="video/mp4,.mp4"
                onChange={(event) => acceptFile(event.target.files?.[0])}
              />
            </section>

            <section className="rounded-lg border border-[#cfd8d4] bg-white p-4 shadow-sm">
              <h2 className="text-sm font-semibold text-[#17211d]">Run controls</h2>

              <div className="mt-4 grid gap-2">
                <button
                  type="button"
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-[#0a5c5a] px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-[#084d4b] disabled:cursor-not-allowed disabled:bg-[#a8b8b3]"
                  onClick={processClip}
                  disabled={!file || isUploading}
                  title="Process selected clip"
                >
                  {isUploading ? (
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  ) : (
                    <Play className="h-4 w-4" aria-hidden="true" />
                  )}
                  {isUploading ? 'Sending' : 'Process'}
                </button>

                <button
                  type="button"
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-[#c7d0cc] bg-white px-4 text-sm font-semibold text-[#34423e] transition hover:bg-[#f4f7f5]"
                  onClick={clearProject}
                  title="Clear current clip and stream"
                >
                  <Eraser className="h-4 w-4" aria-hidden="true" />
                  Clear
                </button>
              </div>

              <div className="mt-5">
                <div className="mb-2 flex items-center justify-between text-xs text-[#65716d]">
                  <span>{status === 'streaming' ? 'Ready' : 'Upload'}</span>
                  <span>{progress}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-[#dce5e1]">
                  <div
                    className="h-full rounded-full bg-[#c98525] transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              {error && (
                <div className="mt-4 rounded-lg border border-[#e5b2a8] bg-[#fff3f0] px-3 py-2 text-sm text-[#8f2f20]">
                  {error}
                </div>
              )}
            </section>

            <section className="rounded-lg border border-[#cfd8d4] bg-[#fbfcfb] p-4 text-xs leading-5 text-[#65716d]">
              <div className="mb-2 flex items-center gap-2 font-semibold text-[#34423e]">
                <Gauge className="h-4 w-4 text-[#c98525]" aria-hidden="true" />
                Pipeline
              </div>
              Smoothed frame-pair motion estimation, adaptive vector filtering, and query
              indexing.
            </section>
          </aside>

          <section className="flex min-w-0 flex-col gap-4">
            <div className="grid gap-4 xl:grid-cols-2">
              <article className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-[#cfd8d4] bg-white shadow-sm">
                <div className="flex h-14 items-center justify-between border-b border-[#e2e8e5] px-4">
                  <div>
                    <h2 className="text-sm font-semibold text-[#17211d]">Original</h2>
                    <p className="text-xs text-[#65716d]">Uploaded MP4 source</p>
                  </div>
                  <span className="rounded-md bg-[#f1f5f3] px-2 py-1 text-xs text-[#65716d]">
                    Input
                  </span>
                </div>

                <div className="grid flex-1 place-items-center bg-[#111816] p-3">
                  {originalUrl ? (
                    <video
                      className="aspect-video w-full rounded-lg bg-black object-contain"
                      controls
                      playsInline
                      src={originalUrl}
                    />
                  ) : (
                    <div className="grid aspect-video w-full place-items-center rounded-lg border border-[#2c3733] bg-[#18201d] text-sm text-[#9aa8a3]">
                      No source clip loaded
                    </div>
                  )}
                </div>
              </article>

              <article className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-[#cfd8d4] bg-white shadow-sm">
                <div className="flex h-14 items-center justify-between border-b border-[#e2e8e5] px-4">
                  <div>
                    <h2 className="text-sm font-semibold text-[#17211d]">Motion Field</h2>
                    <p className="text-xs text-[#65716d]">Sparse high-motion vectors</p>
                  </div>
                  <span className="rounded-md bg-[#edf7f3] px-2 py-1 text-xs text-[#0a5c5a]">
                    Output
                  </span>
                </div>

                <div className="grid flex-1 place-items-center bg-[#111816] p-3">
                  {streamUrl ? (
                    <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-black">
                      <img
                        key={streamUrl}
                        src={streamUrl}
                        alt="Live optical flow vector stream"
                        className="h-full w-full object-contain"
                        onLoad={() => setStreamError(null)}
                        onError={() => {
                          setStreamError('The stream failed to render. Restart processing or check the backend log.')
                          setStatus('error')
                        }}
                      />
                      {streamError && (
                        <div className="absolute inset-0 grid place-items-center bg-black/80 p-6 text-center text-sm text-[#f2c1b6]">
                          {streamError}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="grid aspect-video w-full place-items-center rounded-lg border border-[#2c3733] bg-[linear-gradient(135deg,#18201d,#111816)] text-center text-sm text-[#9aa8a3]">
                      Process a clip to view filtered motion vectors
                    </div>
                  )}
                </div>
              </article>
            </div>

            <section className="rounded-lg border border-[#cfd8d4] bg-white shadow-sm">
              <div className="grid gap-4 border-b border-[#e2e8e5] p-4 xl:grid-cols-[1fr_auto] xl:items-end">
                <div className="flex items-start gap-3">
                  <div className="grid h-10 w-10 place-items-center rounded-lg bg-[#edf7f3] text-[#0a5c5a]">
                    <Search className="h-5 w-5" aria-hidden="true" />
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold text-[#17211d]">Action search</h2>
                    <p className="mt-1 text-xs leading-5 text-[#65716d]">
                      Query the motion index and inspect representative frames for each segment.
                    </p>
                  </div>
                </div>

                <div className="grid gap-2 sm:grid-cols-[140px_160px_auto]">
                  <label className="grid gap-1 text-xs font-semibold text-[#34423e]">
                    Direction
                    <select
                      className="h-10 rounded-lg border border-[#c7d0cc] bg-white px-3 text-sm font-medium text-[#17211d]"
                      value={searchDirection}
                      onChange={(event) => setSearchDirection(event.target.value as MotionDirection)}
                    >
                      {directions.map((direction) => (
                        <option key={direction} value={direction}>
                          {direction}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="grid gap-1 text-xs font-semibold text-[#34423e]">
                    Minimum velocity
                    <input
                      className="h-10 rounded-lg border border-[#c7d0cc] bg-white px-3 text-sm font-medium text-[#17211d]"
                      type="number"
                      min="0"
                      step="0.5"
                      value={minVelocity}
                      onChange={(event) => setMinVelocity(Number(event.target.value))}
                    />
                  </label>

                  <button
                    type="button"
                    className="mt-5 inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-[#0a5c5a] px-5 text-sm font-semibold text-white transition hover:bg-[#084d4b] disabled:cursor-not-allowed disabled:opacity-60 sm:mt-auto"
                    onClick={searchActions}
                    disabled={isSearching}
                    title="Search motion index"
                  >
                    {isSearching ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    ) : (
                      <Search className="h-4 w-4" aria-hidden="true" />
                    )}
                    Search
                  </button>
                </div>
              </div>

              {searchError && (
                <div className="m-4 rounded-lg border border-[#e5b2a8] bg-[#fff3f0] px-3 py-2 text-sm text-[#8f2f20]">
                  {searchError}
                </div>
              )}

              <div className="p-4">
                {searchResults ? (
                  <>
                    <div className="mb-4 grid gap-3 sm:grid-cols-3">
                      <div className="rounded-lg border border-[#dbe4e0] bg-[#fbfcfb] px-3 py-3">
                        <p className="text-xs text-[#65716d]">Matching frames</p>
                        <p className="mt-1 text-xl font-semibold text-[#17211d]">
                          {searchResults.match_count}
                        </p>
                      </div>
                      <div className="rounded-lg border border-[#dbe4e0] bg-[#fbfcfb] px-3 py-3">
                        <p className="text-xs text-[#65716d]">Segments</p>
                        <p className="mt-1 text-xl font-semibold text-[#17211d]">
                          {searchResults.segments.length}
                        </p>
                      </div>
                      <div className="rounded-lg border border-[#dbe4e0] bg-[#fbfcfb] px-3 py-3">
                        <p className="text-xs text-[#65716d]">Query</p>
                        <p className="mt-1 truncate text-sm font-semibold text-[#17211d]">
                          {searchResults.query.direction} / {searchResults.query.min_velocity}
                        </p>
                      </div>
                    </div>

                    {searchResults.segments.length === 0 ? (
                      <div className="grid min-h-44 place-items-center rounded-lg border border-dashed border-[#c7d0cc] bg-[#f7faf8] px-6 text-center">
                        <div>
                          <Image className="mx-auto mb-3 h-8 w-8 text-[#8b9994]" aria-hidden="true" />
                          <p className="text-sm font-semibold text-[#34423e]">No visual matches found</p>
                          <p className="mt-1 text-xs text-[#65716d]">
                            Lower the velocity threshold or try another direction.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
                        {searchResults.segments.map((segment) => (
                          <article
                            key={`${segment.start_frame}-${segment.end_frame}`}
                            className="overflow-hidden rounded-lg border border-[#dbe4e0] bg-[#fbfcfb]"
                          >
                            <div className="relative aspect-video bg-[#111816]">
                              {segment.thumbnail_url ? (
                                <img
                                  src={buildMediaUrl(segment.thumbnail_url)}
                                  alt={`Representative frame ${segment.representative_frame}`}
                                  className="h-full w-full object-cover"
                                  loading="lazy"
                                />
                              ) : (
                                <div className="grid h-full place-items-center text-sm text-[#9aa8a3]">
                                  No preview
                                </div>
                              )}
                              <span className="absolute right-2 top-2 rounded-md bg-black/70 px-2 py-1 text-xs font-semibold text-white">
                                {segment.direction}
                              </span>
                            </div>

                            <div className="p-3">
                              <div className="mb-3 flex items-center justify-between gap-3">
                                <span className="inline-flex items-center gap-2 text-sm font-semibold text-[#17211d]">
                                  <Clock3 className="h-4 w-4 text-[#c98525]" aria-hidden="true" />
                                  {formatSeconds(segment.start_timestamp)} - {formatSeconds(segment.end_timestamp)}
                                </span>
                                <span className="text-xs text-[#65716d]">
                                  {segment.frame_count} frames
                                </span>
                              </div>

                              <div className="grid grid-cols-2 gap-2 text-xs text-[#65716d]">
                                <span>Frames {segment.start_frame}-{segment.end_frame}</span>
                                <span>Preview {segment.representative_frame}</span>
                                <span>Mean {segment.mean_velocity.toFixed(2)}</span>
                                <span>Peak {segment.peak_velocity.toFixed(2)}</span>
                              </div>
                            </div>
                          </article>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="grid min-h-52 place-items-center rounded-lg border border-dashed border-[#c7d0cc] bg-[#f7faf8] px-6 text-center">
                    <div>
                      <Search className="mx-auto mb-3 h-8 w-8 text-[#8b9994]" aria-hidden="true" />
                      <p className="text-sm font-semibold text-[#34423e]">No query submitted</p>
                      <p className="mt-1 text-xs text-[#65716d]">
                        Process a clip, then search for directional motion segments.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </section>
          </section>
        </div>
      </section>
    </main>
  )
}
