import React, { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

const API_BASE = import.meta.env.VITE_API_BASE || ''

type Item = {
  article_id: string
  url?: string
  title?: string
  label: 'Fake' | 'Real' | 'Suspect'
  confidence: number
  virality_score: number
  high_priority?: boolean
  created_at?: string
  tags?: string[]
}

type ReviewPayload = {
  article_id: string
  action: 'accept' | 'reject'
  reviewer_id: string
  note?: string
}

function toCSV(items: Item[]): string {
  const headers = ['article_id','title','url','label','confidence','virality_score','high_priority','created_at']
  const rows = items.map(i => headers.map(h => JSON.stringify((i as any)[h] ?? '')))
  return [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
}

const ReviewerQueue: React.FC = () => {
  const [items, setItems] = useState<Item[]>([])
  const [selected, setSelected] = useState<Record<string, boolean>>({})
  const [search, setSearch] = useState('')
  const [tag, setTag] = useState('')
  const [note, setNote] = useState('')
  const reviewerId = 'reviewer-001'

  useEffect(() => {
    let mounted = true
    async function load() {
      // Placeholder: fetch prioritized items
      try {
        const res = await fetch(`${API_BASE}/api/reviewer/queue`)
        if (res.ok) {
          const data = await res.json()
          if (mounted) setItems(data || [])
        }
      } catch {}
    }
    load()
    const t = setInterval(load, 15000)
    return () => { mounted = false; clearInterval(t) }
  }, [])

  const filtered = useMemo(() => {
    let list = items
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(i => (i.title||'').toLowerCase().includes(q) || (i.url||'').toLowerCase().includes(q))
    }
    if (tag.trim()) {
      const q = tag.toLowerCase()
      list = list.filter(i => (i.tags||[]).some(t => t.toLowerCase().includes(q)))
    }
    // Sort: high_priority first, then suspect with high virality, then by time desc
    list = [...list].sort((a,b) => {
      const ap = (a.high_priority?1:0) - (b.high_priority?1:0)
      if (ap !== 0) return -ap
      const as = (a.label==='Suspect' && a.virality_score>0.6) ? 1:0
      const bs = (b.label==='Suspect' && b.virality_score>0.6) ? 1:0
      if (as !== bs) return bs - as
      const at = new Date(a.created_at||0).getTime()
      const bt = new Date(b.created_at||0).getTime()
      return bt - at
    })
    return list
  }, [items, search, tag])

  const allChecked = useMemo(() => filtered.length>0 && filtered.every(i => selected[i.article_id]), [filtered, selected])
  const someChecked = useMemo(() => filtered.some(i => selected[i.article_id]), [filtered, selected])

  function toggleAll() {
    if (allChecked) {
      const next: Record<string, boolean> = { ...selected }
      filtered.forEach(i => { delete next[i.article_id] })
      setSelected(next)
    } else {
      const next: Record<string, boolean> = { ...selected }
      filtered.forEach(i => { next[i.article_id] = true })
      setSelected(next)
    }
  }

  function toggleOne(id: string) {
    setSelected(s => ({ ...s, [id]: !s[id] }))
  }

  async function reviewOne(article_id: string, action: 'accept'|'reject') {
    try {
      await fetch(`${API_BASE}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ article_id, action, reviewer_id: reviewerId, note }) })
    } catch {}
  }

  async function bulkAction(action: 'accept'|'reject') {
    const ids = filtered.filter(i => selected[i.article_id]).map(i => i.article_id)
    await Promise.all(ids.map(id => reviewOne(id, action)))
  }

  function exportCSV() {
    const ids = filtered.filter(i => selected[i.article_id])
    const csv = toCSV(ids)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'reviewer_queue.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search title/url" className="px-3 py-2 rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800" />
        <input value={tag} onChange={e=>setTag(e.target.value)} placeholder="Tag filter" className="px-3 py-2 rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800" />
        <input value={note} onChange={e=>setNote(e.target.value)} placeholder="Add note (applies to actions)" className="flex-1 px-3 py-2 rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800" />
        <button onClick={()=>bulkAction('accept')} disabled={!someChecked} className="px-3 py-2 rounded bg-green-600 text-white disabled:opacity-50">Accept selected</button>
        <button onClick={()=>bulkAction('reject')} disabled={!someChecked} className="px-3 py-2 rounded bg-red-600 text-white disabled:opacity-50">Reject selected</button>
        <button onClick={exportCSV} disabled={!someChecked} className="px-3 py-2 rounded bg-neutral-200 dark:bg-neutral-800">Export CSV</button>
      </div>

      <div className="bg-white dark:bg-neutral-900 rounded-lg shadow">
        <table className="w-full text-sm">
          <thead className="text-left border-b border-neutral-200 dark:border-neutral-800">
            <tr>
              <th className="p-3"><input type="checkbox" aria-checked={someChecked && !allChecked} checked={allChecked} onChange={toggleAll} /></th>
              <th className="p-3">Title</th>
              <th className="p-3">Label</th>
              <th className="p-3">Conf</th>
              <th className="p-3">Virality</th>
              <th className="p-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(i => (
              <tr key={i.article_id} className="border-b border-neutral-100 dark:border-neutral-800">
                <td className="p-3"><input type="checkbox" checked={!!selected[i.article_id]} onChange={()=>toggleOne(i.article_id)} /></td>
                <td className="p-3 max-w-[28rem]">
                  <div className="truncate font-medium"><Link className="hover:underline" to={`/article/${encodeURIComponent(i.article_id)}`}>{i.title || i.url}</Link></div>
                  <div className="text-xs text-gray-500 truncate">{i.url}</div>
                </td>
                <td className="p-3">{i.label}</td>
                <td className="p-3">{Math.round(i.confidence*100)}%</td>
                <td className="p-3">{Math.round(i.virality_score*100)}</td>
                <td className="p-3 flex items-center gap-2">
                  <button onClick={()=>reviewOne(i.article_id,'accept')} className="px-2 py-1 rounded bg-green-600 text-white">Accept</button>
                  <button onClick={()=>reviewOne(i.article_id,'reject')} className="px-2 py-1 rounded bg-red-600 text-white">Reject</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default ReviewerQueue
