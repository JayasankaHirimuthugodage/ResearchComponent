import type { ActivityResponse } from '../../api'
import { DOMAIN_LABELS } from '../../app/constants'

export type ActivityCard = {
  title: string
  instructions: string
  domains: string[]
}

export function buildActivityCards(activities: ActivityResponse): ActivityCard[] {
  const unique = new Map<string, ActivityCard>()

  for (const domain of activities.domains) {
    const label = DOMAIN_LABELS[domain.domain] ?? domain.domain
    for (const activity of domain.activities) {
      const key = `${activity.title}\u0000${activity.instructions}`
      const existing = unique.get(key)
      if (existing) {
        if (!existing.domains.includes(label)) existing.domains.push(label)
      } else {
        unique.set(key, { ...activity, domains: [label] })
      }
    }
  }

  return Array.from(unique.values())
}

