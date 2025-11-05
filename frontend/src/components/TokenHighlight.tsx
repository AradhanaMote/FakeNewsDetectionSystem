import React from 'react'

type Token = { token: string; weight: number }

type Props = {
  text: string
  tokens: Token[]
}

function normalizeWeights(tokens: Token[]): Record<string, number> {
  let max = 0
  for (const t of tokens) max = Math.max(max, Math.abs(t.weight))
  const map: Record<string, number> = {}
  for (const t of tokens) {
    const key = t.token.toLowerCase()
    map[key] = Math.max(map[key] || 0, Math.abs(t.weight) / (max || 1))
  }
  return map
}

const TokenHighlight: React.FC<Props> = ({ text, tokens }) => {
  const weightMap = normalizeWeights(tokens)
  const words = text.split(/(\s+)/)

  return (
    <div role="article" aria-label="explainable text">
      <p className="leading-7 text-gray-900 dark:text-gray-100 text-sm sm:text-base">
        {words.map((w, idx) => {
          const key = w.trim().toLowerCase()
          const weight = weightMap[key] || 0
          if (!w.trim()) return <span key={idx}>{w}</span>
          const bg = `rgba(245, 158, 11, ${Math.min(0.8, weight)})` // amber-500
          return (
            <span
              key={idx}
              style={{ backgroundColor: weight > 0 ? bg : undefined, borderRadius: 3, paddingInline: weight > 0 ? 1 : 0 }}
              title={weight > 0 ? `weight=${weight.toFixed(2)}` : undefined}
            >
              {w}
            </span>
          )
        })}
      </p>
    </div>
  )
}

export default TokenHighlight
