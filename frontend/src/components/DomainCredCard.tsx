import React from 'react'

type Props = {
  domain: string
  trustScore: number  // 0..100
  publishedCount: number
  topTags: string[]
}

const DomainCredCard: React.FC<Props> = ({ domain, trustScore, publishedCount, topTags }) => {
  return (
    <section className="bg-white dark:bg-neutral-900 rounded-lg shadow p-4" aria-labelledby="domain-cred">
      <h3 id="domain-cred" className="text-sm font-semibold mb-2">Domain reputation</h3>
      <div className="text-sm flex items-center gap-2">
        <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-neutral-800">{domain}</span>
        <span className="text-xs text-gray-500">published: {publishedCount}</span>
      </div>
      <div className="mt-3">
        <div className="text-xs text-gray-500 mb-1">Trust</div>
        <div className="h-2 bg-gray-200 rounded w-full">
          <div className="h-2 bg-green-500 rounded" style={{ width: `${trustScore}%` }} aria-label={`trust ${trustScore} percent`} role="img" />
        </div>
      </div>
      <div className="mt-3">
        <div className="text-xs text-gray-500 mb-1">Top tags</div>
        <div className="flex flex-wrap gap-1">
          {topTags.map(t => (
            <span key={t} className="px-2 py-0.5 rounded bg-gray-100 dark:bg-neutral-800 text-xs">{t}</span>
          ))}
        </div>
      </div>
    </section>
  )
}

export default DomainCredCard
