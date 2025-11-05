import React, { useEffect, useMemo, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import TokenHighlight from '../components/TokenHighlight'

const API_BASE = import.meta.env.VITE_API_BASE || ''

type Prediction = {
  label: 'Fake' | 'Real' | 'Suspect'
  confidence: number
  explanation_tokens: [string, number][]
  virality_score: number
  model_version: string
}

type Article = {
  id: string
  url: string
  title: string
  text: string
  source?: string
  published_at?: string
}

const Sparkline: React.FC<{ data: number[] }> = ({ data }) => {
  const width = 160
  const height = 40
  const points = data.length
  const max = Math.max(1, ...data)
  const d = data
    .map((v, i) => {
      const x = (i / Math.max(1, points - 1)) * width
      const y = height - (v / max) * height
      return `${x},${y}`
    })
    .join(' ')
  return (
    <svg width={width} height={height} aria-label="sparkline">
      <polyline fill="none" stroke="#8884d8" strokeWidth="2" points={d} />
    </svg>
  )
}

const Gauge: React.FC<{ value: number }> = ({ value }) => {
  const pct = Math.round(value * 100)
  const color = value > 0.7 ? '#e74c3c' : value > 0.4 ? '#f39c12' : '#2ecc71'
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg viewBox="0 0 36 36" className="w-full h-full">
          <path className="text-gray-200" stroke="currentColor" strokeWidth="3" fill="none" d="M18 2a16 16 0 1 1 0 32 16 16 0 1 1 0-32" />
          <path stroke={color} strokeWidth="3" fill="none" d={`M18 2a16 16 0 1 1 0 32 16 16 0 1 1 0-32`} strokeDasharray={`${pct}, 100`} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center text-sm font-semibold">{pct}%</div>
      </div>
      <div className="text-xs text-gray-500">Virality</div>
    </div>
  )
}

const Skeleton: React.FC<{ lines?: number }> = ({ lines = 3 }) => (
  <div className="animate-pulse space-y-2">
    {Array.from({ length: lines }).map((_, i) => (
      <div key={i} className="h-4 bg-gray-200 dark:bg-neutral-800 rounded" />
    ))}
  </div>
)

const ArticleDetail: React.FC = () => {
  const { id } = useParams()
  const navigate = useNavigate()
  const [article, setArticle] = useState<Article | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(true)
  const [note, setNote] = useState('')

  useEffect(() => {
    let mounted = true
    async function fetchAll() {
      setLoading(true)
      try {
        const [aRes, pRes] = await Promise.all([
          fetch(`${API_BASE}/api/article/${id}`),
          fetch(`${API_BASE}/api/prediction/${id}`),
        ])
        if (!mounted) return
        if (aRes.ok) setArticle(await aRes.json())
        if (pRes.ok) setPrediction(await pRes.json())
      } finally {
        if (mounted) setLoading(false)
      }
    }
    fetchAll()
    return () => {
      mounted = false
    }
  }, [id])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'a') handleAction('accept')
      if (e.key === 'r') handleAction('reject')
      if (e.key === 'Escape') navigate(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [article])

  const tokens = useMemo(() => {
    const arr = prediction?.explanation_tokens || []
    // explanation_tokens format from backend is [token, weight]
    return arr.map(([token, weight]) => ({ token, weight }))
  }, [prediction])

  function handleAction(kind: 'accept' | 'reject') {
    // Placeholder: post to review endpoint
    console.log('review', kind, id, note)
  }

  const rightPanel = (
    <aside className="w-full md:w-80 flex-shrink-0">
      <div className="bg-white dark:bg-neutral-900 rounded-lg shadow p-4 space-y-4">
        <div>
          <div className="text-sm font-semibold mb-2">Domain reputation</div>
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
            <span>{new URL(article?.url || 'https://example.com').hostname.replace('www.', '')}</span>
            <span className="px-2 py-0.5 text-xs rounded bg-gray-100 dark:bg-neutral-800">rep: n/a</span>
          </div>
        </div>
        <div>
          <div className="text-sm font-semibold mb-2">Mentions timeline</div>
          <Sparkline data={[1,2,3,2,4,5,3,6,5,7,6,8]} />
        </div>
        <div>
          <div className="text-sm font-semibold mb-2">Virality</div>
          <Gauge value={prediction?.virality_score || 0} />
        </div>
        <div>
          <div className="text-sm font-semibold mb-2">Fact-check</div>
          <div className="text-sm text-gray-600 dark:text-gray-300">No matches</div>
        </div>
      </div>
    </aside>
  )

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-4 grid md:grid-cols-[1fr,20rem] gap-4">
        <div className="space-y-4">
          <Skeleton lines={2} />
          <Skeleton lines={8} />
        </div>
        <Skeleton lines={12} />
      </div>
    )
  }

  if (!article) {
    return <div className="p-4">Not found</div>
  }

  return (
    <div className="max-w-6xl mx-auto p-4 grid md:grid-cols-[1fr,20rem] gap-4">
      <div className="space-y-4">
        <div className="bg-white dark:bg-neutral-900 rounded-lg shadow p-4">
          <div className="flex items-start justify-between gap-4">
            <h2 className="text-2xl font-semibold">{article.title}</h2>
            {prediction && (
              <div className="text-right">
                <div className="text-sm">{prediction.label}</div>
                <div className="text-xs text-gray-500">conf {Math.round((prediction.confidence||0)*100)}%</div>
              </div>
            )}
          </div>
          <div className="mt-2 text-sm text-gray-500">
            <a className="hover:underline" href={article.url} target="_blank" rel="noreferrer">{article.url}</a>
          </div>
        </div>
        <div className="bg-white dark:bg-neutral-900 rounded-lg shadow p-4">
          {prediction ? (
            <TokenHighlight text={article.text} tokens={tokens} />
          ) : (
            <p>{article.text}</p>
          )}
        </div>
        <div className="bg-white dark:bg-neutral-900 rounded-lg shadow p-4 flex items-center gap-2">
          <button onClick={() => handleAction('accept')} className="px-3 py-2 rounded bg-green-600 text-white hover:bg-green-700 transition">Mark as Reviewed: Accept (A)</button>
          <button onClick={() => handleAction('reject')} className="px-3 py-2 rounded bg-red-600 text-white hover:bg-red-700 transition">Mark as Reviewed: Reject (R)</button>
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Add note" className="flex-1 px-3 py-2 rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800" />
          <button className="px-3 py-2 rounded bg-neutral-200 dark:bg-neutral-800 hover:bg-neutral-300 dark:hover:bg-neutral-700 transition" onClick={() => navigator.share?.({ title: article.title, url: article.url }).catch(()=>{})}>Share</button>
        </div>
      </div>
      {rightPanel}
    </div>
  )
}

export default ArticleDetail
