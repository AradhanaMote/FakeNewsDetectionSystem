import React from 'react'

type Props = { value: number }

const ViralityGauge: React.FC<Props> = ({ value }) => {
  const pct = Math.round(value * 100)
  const gradientId = 'viralityGrad'
  return (
    <div className="flex flex-col items-center" role="img" aria-label={`virality ${pct} percent`}>
      <svg viewBox="0 0 36 36" className="w-24 h-24" title={`${pct}%`}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22c55e" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        <path stroke="#e5e7eb" strokeWidth="3" fill="none" d="M18 2a16 16 0 1 1 0 32 16 16 0 1 1 0-32" />
        <path stroke={`url(#${gradientId})`} strokeWidth="3" fill="none" d={`M18 2a16 16 0 1 1 0 32 16 16 0 1 1 0-32`} strokeDasharray={`${pct}, 100`} />
      </svg>
      <div className="text-sm font-semibold">{pct}%</div>
      <div className="text-xs text-gray-500">Virality</div>
    </div>
  )
}

export default ViralityGauge
