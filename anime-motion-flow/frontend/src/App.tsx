import { useCallback, useMemo, useState } from 'react'
import axios from 'axios'
import { Film, Loader2, UploadCloud } from 'lucide-react'

type Metric = {
  frame: number
  mean_flow: number
  max_flow: number
  line_density: number
}

type UploadResponse = {
  upload_id: string
  filename: string
  frames_read: number
  fps: number
  torch_device: string
  metrics: Metric[]
  previews: string[]
  status: string
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function formatNumber(value: number | undefined) {
  return typeof value === 'number' ? value.toLocaleString() : 'n/a'
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const latestMetric = useMemo(() => result?.metrics.at(-1), [result])

  const acceptFile = useCallback((candidate?: File) => {
    if (!candidate) {
      return
    }

    const isMp4 =
      candidate.type === 'video/mp4' || candidate.name.toLowerCase().endsWith('.mp4')

    if (!isMp4) {
      setError('Please select an MP4 video.')
      return
    }

    setFile(candidate)
    setResult(null)
    setError(null)
    setProgress(0)
  }, [])

  const upload = async () => {
    if (!file) {
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    setIsUploading(true)
    setError(null)

    try {
      const response = await axios.post<UploadResponse>(
        `${API_URL}/upload-video`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (event) => {
            if (!event.total) {
              return
            }
            setProgress(Math.round((event.loaded / event.total) * 100))
          },
        },
      )

      setResult(response.data)
    } catch (err) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail ?? err.message
        : 'Upload failed'
      setError(String(message))
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-5 py-8">
        <header className="flex flex-col gap-2 border-b border-zinc-800 pb-5">
          <div className="flex items-center gap-3">
            <Film className="h-7 w-7 text-cyan-300" aria-hidden="true" />
            <h1 className="text-2xl font-semibold tracking-normal">
              Self-Supervised Cel-Anime Motion Flow
            </h1>
          </div>
          <p className="max-w-3xl text-sm leading-6 text-zinc-400">
            Upload an MP4 to run frame decoding, line-art alignment previews, and
            the placeholder optical-flow pass.
          </p>
        </header>

        <div
          className={[
            'flex min-h-64 flex-col items-center justify-center rounded border border-dashed p-8 text-center transition',
            isDragging ? 'border-cyan-300 bg-cyan-950/30' : 'border-zinc-700 bg-zinc-900',
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
          <UploadCloud className="mb-4 h-10 w-10 text-cyan-300" aria-hidden="true" />
          <label className="cursor-pointer rounded bg-cyan-400 px-4 py-2 text-sm font-semibold text-zinc-950 hover:bg-cyan-300">
            Choose MP4
            <input
              className="sr-only"
              type="file"
              accept="video/mp4,.mp4"
              onChange={(event) => acceptFile(event.target.files?.[0])}
            />
          </label>
          <p className="mt-3 max-w-full break-words text-sm text-zinc-400">
            {file ? file.name : 'Drop a video here or select one from disk.'}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            className="inline-flex min-h-10 items-center gap-2 rounded bg-zinc-100 px-4 py-2 text-sm font-semibold text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!file || isUploading}
            onClick={upload}
          >
            {isUploading ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <UploadCloud className="h-4 w-4" aria-hidden="true" />
            )}
            Upload and Process
          </button>

          {isUploading && (
            <div className="h-2 w-56 overflow-hidden rounded bg-zinc-800" aria-label="Upload progress">
              <div className="h-full bg-cyan-300 transition-all" style={{ width: `${progress}%` }} />
            </div>
          )}

          {error && <span className="text-sm text-red-300">{error}</span>}
        </div>

        {result && (
          <section className="grid gap-5 lg:grid-cols-[280px_1fr]">
            <aside className="rounded border border-zinc-800 bg-zinc-900 p-4">
              <h2 className="mb-3 text-sm font-semibold text-zinc-300">Run Summary</h2>
              <dl className="grid gap-2 text-sm">
                <div className="flex justify-between gap-4">
                  <dt className="text-zinc-500">Frames</dt>
                  <dd>{formatNumber(result.frames_read)}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-zinc-500">FPS</dt>
                  <dd>{formatNumber(result.fps)}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-zinc-500">Torch</dt>
                  <dd>{result.torch_device}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-zinc-500">Mean flow</dt>
                  <dd>{formatNumber(latestMetric?.mean_flow)}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-zinc-500">Line density</dt>
                  <dd>{formatNumber(latestMetric?.line_density)}</dd>
                </div>
              </dl>
            </aside>

            <div className="rounded border border-zinc-800 bg-zinc-900 p-4">
              <h2 className="mb-3 text-sm font-semibold text-zinc-300">
                Simulated Output Stream
              </h2>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
                {result.previews.map((preview, index) => (
                  <img
                    key={`${result.upload_id}-${index}`}
                    src={preview}
                    alt={`Processed preview ${index + 1}`}
                    className="aspect-video w-full rounded object-cover"
                  />
                ))}
              </div>
            </div>
          </section>
        )}
      </section>
    </main>
  )
}
