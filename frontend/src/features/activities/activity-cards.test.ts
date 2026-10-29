import { describe, expect, it } from 'vitest'
import type { ActivityResponse } from '../../api'
import { buildActivityCards } from './activity-cards'

describe('buildActivityCards', () => {
  it('deduplicates identical activities and preserves their domain labels', () => {
    const activities = {
      domains: [
        { domain: 'communication', activities: [{ title: 'Shared play', instructions: 'Take turns.' }] },
        { domain: 'cognitive', activities: [{ title: 'Shared play', instructions: 'Take turns.' }] },
      ],
    } as ActivityResponse

    expect(buildActivityCards(activities)).toEqual([
      {
        title: 'Shared play',
        instructions: 'Take turns.',
        domains: ['Communication', 'Cognitive development'],
      },
    ])
  })

  it('keeps activities with different instructions separate', () => {
    const activities = {
      domains: [{
        domain: 'social_emotional',
        activities: [
          { title: 'Talk together', instructions: 'Name feelings.' },
          { title: 'Talk together', instructions: 'Take turns.' },
        ],
      }],
    } as ActivityResponse

    expect(buildActivityCards(activities)).toHaveLength(2)
  })
})

