import React from 'react'

export type ArticleSummary = {
  article_id: string
  title: string
  snippet?: string
  url?: string
  domain?: string
  label: 'Fake' | 'Real' | 'Suspect'
  confidence: number
  virality_score: number
  published_at?: string
}

type Props = {
  item: ArticleSummary
  onShare?: (item: ArticleSummary) => void
  onOpen?: (item: ArticleSummary) => void
}

const ConfidenceRibbon: React.FC<{ confidence: number }> = ({ confidence }) => {
  const pct = Math.round((confidence || 0) * 100)
  const gradient = 'bg-gradient-to-r from-red-500 via-yellow-500 to-green-500'
  return (
    <div aria-label={`confidence ${pct} percent`} className="flex items-center gap-2" role="img">
      <div className={`h-2 w-28 bg-gray-200 rounded overflow-hidden`}>
        <div className={`h-2 ${gradient}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-600" aria-hidden>{pct}%</span>
    </div>
  )
}

const ViralityBadge: React.FC<{ score: number }> = ({ score }) => {
  const pulse = score > 0.7 ? 'animate-pulse' : ''
  const color = score > 0.7 ? 'bg-red-500' : score > 0.4 ? 'bg-yellow-500' : 'bg-gray-300'
  const label = `virality ${Math.round(score * 100)}`
  return <span role="img" aria-label={label} className={`px-2 py-0.5 rounded text-white text-xs ${color} ${pulse}`}>V {Math.round(score * 100)}</span>
}

const ArticleCard: React.FC<Props> = ({ item, onShare, onOpen }) => {
  const domain = item.domain || (() => {
    try { return item.url ? new URL(item.url).hostname.replace('www.', '') : '' } catch { return '' }
  })()
  return (
    <article className="bg-white dark:bg-neutral-900 rounded-lg shadow p-4 flex flex-col gap-2" aria-labelledby={`t-${item.article_id}`}>
      <div className="flex items-start justify-between gap-2">
        <h3 id={`t-${item.article_id}`} className="text-base sm:text-lg font-semibold">
          <button className="text-left hover:underline" onClick={() => onOpen?.(item)} aria-label={`open ${item.title}`}>{item.title}</button>
        </h3>
        <span className="px-2 py-0.5 rounded text-white text-xs "
          style={{ backgroundColor: item.label==='Real'?'#2ecc71':item.label==='Fake'?'#e74c3c':'#f39c12' }}>{item.label}</span>
      </div>
      {item.snippet && <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-3">{item.snippet}</p>}
      <div className="flex items-center justify-between mt-1">
        <div className="flex items-center gap-3">
          {domain && <span className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-neutral-800" aria-label={`domain ${domain}`}>{domain}</span>}
          <ConfidenceRibbon confidence={item.confidence} />
        </div>
        <ViralityBadge score={item.virality_score} />
      </div>
      <div className="flex items-center justify-between text-xs text-gray-500 mt-1">
        <span>{item.published_at ? new Date(item.published_at).toLocaleString() : ''}</span>
        <div className="flex items-center gap-2">
          {item.url && <a className="px-2 py-1 rounded bg-neutral-100 dark:bg-neutral-800 hover:underline" href={item.url} target="_blank" rel="noreferrer">Open</a>}
          <button className="px-2 py-1 rounded bg-neutral-100 dark:bg-neutral-800" onClick={() => onShare?.(item)}>Share</button>
        </div>
      </div>
    </article>
  )
}

export default ArticleCard
