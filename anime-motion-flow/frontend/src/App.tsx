import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  CircleAlert,
  Eraser,
  FileVideo,
  Loader2,
  Play,
  UploadCloud,
} from 'lucide-react'

type ProcessVideoResponse = {
  job_id: string
  filename: string
  stream_url: string
  status: string
}

type RunState = 'idle' | 'ready' | 'uploading' | 'streaming' | 'error'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const statusCopy: Record<RunState, string> = {
  idle: 'Awaiting clip',
  ready: 'Ready to process',
  uploading: 'Uploading source',
  streaming: 'Streaming vectors',
  error: 'Action needed',
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function getErrorMessage(err: unknown) {
  if (axios.isAxiosError(err)) {
    if (err.code === 'ECONNABORTED') {
      return 'The backend did not respond. Check that FastAPI is running on port 8000.'
    }
    const detail = err.response?.data?.detail
    return typeof detail === 'string' ? detail : err.message
  }
  return 'The video could not be processed.'
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [originalUrl, setOriginalUrl] = useState<string | null>(null)
  const [streamUrl, setStreamUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<RunState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [streamError, setStreamError] = useState<string | null>(null)
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
    setOriginalUrl(URL.createObjectURL(candidate))
    setStreamUrl(null)
    setStreamError(null)
    setProgress(0)
    setError(null)
    setStatus('ready')
  }

  const clearProject = () => {
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    setFile(null)
    setOriginalUrl(null)
    setStreamUrl(null)
    setStreamError(null)
    setIsDragging(false)
    setIsUploading(false)
    setProgress(0)
    setStatus('idle')
    setError(null)
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

      const nextStreamUrl = response.data.stream_url.startsWith('http')
        ? response.data.stream_url
        : `${API_URL}${response.data.stream_url}`
      const cacheKey = Date.now()
      const liveStreamUrl = `${nextStreamUrl}${nextStreamUrl.includes('?') ? '&' : '?'}t=${cacheKey}`

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

  return (
    <main className="min-h-screen bg-[#eef2f0] text-[#18201d]">
      <div className="fixed inset-0 -z-10 bg-[linear-gradient(135deg,rgba(8,86,82,0.10)_0%,rgba(238,242,240,0)_36%),radial-gradient(circle_at_85%_18%,rgba(201,133,37,0.18),rgba(238,242,240,0)_28%)]" />

      <section className="mx-auto flex min-h-screen w-full max-w-[1440px] flex-col px-4 py-4 md:px-6 lg:px-8">
        <header className="grid gap-4 border-b border-[#c7d0cc] pb-4 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <div className="mb-3 flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-lg bg-[#0a5c5a] text-white shadow-sm">
                <Activity className="h-5 w-5" aria-hidden="true" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#67736f]">
                  Cel-animation motion lab
                </p>
                <h1 className="text-2xl font-semibold tracking-normal text-[#141917] md:text-3xl">
                  Anime Motion Flow
                </h1>
              </div>
            </div>
            <p className="max-w-3xl text-sm leading-6 text-[#53605b]">
              Compare an uploaded MP4 against a live structural line-art flow stream.
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

        <div className="grid flex-1 gap-4 py-4 lg:grid-cols-[340px_1fr]">
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
                  <h2 className="text-sm font-semibold text-[#18201d]">Source clip</h2>
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
                <span className="text-sm font-semibold text-[#18201d]">
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
              <h2 className="text-sm font-semibold text-[#18201d]">Run controls</h2>

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
                <ArrowRight className="h-4 w-4 text-[#c98525]" aria-hidden="true" />
                Pipeline
              </div>
              Bilateral smoothing, Sobel line extraction, frame-pair motion estimation,
              and vector overlay streaming.
            </section>
          </aside>

          <section className="grid min-h-[520px] gap-4 xl:grid-cols-2">
            <article className="flex min-w-0 flex-col rounded-lg border border-[#cfd8d4] bg-white shadow-sm">
              <div className="flex h-14 items-center justify-between border-b border-[#e2e8e5] px-4">
                <div>
                  <h2 className="text-sm font-semibold text-[#18201d]">Original</h2>
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

            <article className="flex min-w-0 flex-col rounded-lg border border-[#cfd8d4] bg-white shadow-sm">
              <div className="flex h-14 items-center justify-between border-b border-[#e2e8e5] px-4">
                <div>
                  <h2 className="text-sm font-semibold text-[#18201d]">Motion Field</h2>
                  <p className="text-xs text-[#65716d]">Live vector stream</p>
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
                    Process a clip to view line-art vectors
                  </div>
                )}
              </div>
            </article>
          </section>
        </div>
      </section>
    </main>
  )
}
