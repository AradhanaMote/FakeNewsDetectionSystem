import React, { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || ''

type FeedItem = {
  article_id: string
  title: string
  snippet: string
  label: 'Fake' | 'Real' | 'Suspect'
  confidence: number
  virality_score: number
  published_at?: string
  url?: string
}

type FilterState = {
  label: 'All' | 'Fake' | 'Real' | 'Suspect'
  minConfidence: number
  timeRangeMins: number
}

const LabelChip: React.FC<{ label: FeedItem['label'] }> = ({ label }) => {
  const color = label === 'Real' ? 'bg-real' : label === 'Fake' ? 'bg-fake' : 'bg-suspect'
  return <span className={`px-2 py-0.5 rounded text-white text-xs ${color}`}>{label}</span>
}

const ConfidenceRibbon: React.FC<{ confidence: number }> = ({ confidence }) => {
  const pct = Math.round(confidence * 100)
  const bg = confidence >= 0.7 ? 'bg-green-500' : confidence >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 bg-gray-200 rounded">
        <div className={`h-2 ${bg} rounded`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-600">{pct}%</span>
    </div>
  )
}

const ViralityBadge: React.FC<{ score: number }> = ({ score }) => {
  const pulse = score > 0.7 ? 'animate-pulse' : ''
  const bg = score > 0.7 ? 'bg-red-500' : score > 0.4 ? 'bg-yellow-500' : 'bg-gray-300'
  return <span className={`px-2 py-0.5 rounded text-white text-xs ${bg} ${pulse}`}>V {Math.round(score * 100)}</span>
}

const ArticleCard: React.FC<{ item: FeedItem }> = ({ item }) => {
  const domain = useMemo(() => {
    try {
      if (!item.url) return ''
      const u = new URL(item.url)
      return u.hostname.replace('www.', '')
    } catch {
      return ''
    }
  }, [item.url])

  return (
    <div className="bg-white dark:bg-neutral-900 rounded-lg shadow p-4 flex flex-col gap-2">
      <div className="flex items-start justify-between gap-2">
        <a href={item.url} target="_blank" className="text-lg font-semibold hover:underline">
          {item.title}
        </a>
        <LabelChip label={item.label} />
      </div>
      <div className="text-sm text-gray-600 dark:text-gray-300 line-clamp-3">{item.snippet}</div>
      <div className="flex items-center justify-between mt-2">
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">{domain}</span>
          <ConfidenceRibbon confidence={item.confidence} />
        </div>
        <ViralityBadge score={item.virality_score} />
      </div>
      {item.published_at && (
        <div className="text-xs text-gray-400 mt-1">{new Date(item.published_at).toLocaleString()}</div>
      )}
    </div>
  )
}

const LiveFeed: React.FC = () => {
  const [items, setItems] = useState<FeedItem[]>([])
  const [hasMore, setHasMore] = useState(true)
  const [page, setPage] = useState(0)
  const loaderRef = useRef<HTMLDivElement | null>(null)

  const [filters, setFilters] = useState<FilterState>({ label: 'All', minConfidence: 0.0, timeRangeMins: 720 })

  useEffect(() => {
    const src = new EventSource(`${API_BASE}/stream`)
    src.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        if (!data.article_id) return
        setItems((prev) => [data, ...prev])
      } catch {}
    }
    src.onerror = () => {
      src.close()
    }
    return () => src.close()
  }, [])

  const filtered = useMemo(() => {
    const cutoff = Date.now() - filters.timeRangeMins * 60 * 1000
    return items.filter((it) => {
      if (filters.label !== 'All' && it.label !== filters.label) return false
      if (it.confidence < filters.minConfidence) return false
      if (it.published_at) {
        const t = new Date(it.published_at).getTime()
        if (!isNaN(t) && t < cutoff) return false
      }
      return true
    })
  }, [items, filters])

  // Infinite scroll sentinel
  useEffect(() => {
    const el = loaderRef.current
    if (!el) return
    const obs = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && hasMore) {
        setPage((p) => p + 1)
        // Placeholder: could fetch older items from REST API
        if (items.length > 1000) setHasMore(false)
      }
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [loaderRef.current, hasMore, items.length])

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neutral-950">
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-neutral-900/80 backdrop-blur border-b border-neutral-200 dark:border-neutral-800">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
          <h1 className="text-xl font-bold">Real-Time Fake News Dashboard</h1>
          <div className="flex items-center gap-3">
            <select
              aria-label="Label filter"
              className="px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-sm"
              value={filters.label}
              onChange={(e) => setFilters((f) => ({ ...f, label: e.target.value as FilterState['label'] }))}
            >
              <option>All</option>
              <option>Fake</option>
              <option>Real</option>
              <option>Suspect</option>
            </select>
            <div className="flex items-center gap-2">
              <label className="text-xs">Min conf</label>
              <input
                aria-label="Minimum confidence"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={filters.minConfidence}
                onChange={(e) => setFilters((f) => ({ ...f, minConfidence: parseFloat(e.target.value) }))}
              />
              <span className="text-xs w-8 text-right">{Math.round(filters.minConfidence * 100)}%</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs">Time</label>
              <select
                aria-label="Time range"
                className="px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 text-sm"
                value={filters.timeRangeMins}
                onChange={(e) => setFilters((f) => ({ ...f, timeRangeMins: parseInt(e.target.value) }))}
              >
                <option value={60}>1h</option>
                <option value={180}>3h</option>
                <option value={720}>12h</option>
                <option value={1440}>24h</option>
              </select>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-4">
        <div className="grid gap-3">
          {filtered.map((it) => (
            <ArticleCard key={`${it.article_id}-${it.published_at || ''}`} item={it} />
          ))}
        </div>
        <div ref={loaderRef} className="h-10" />
      </main>
    </div>
  )
}

export default LiveFeed
