import { describe, expect, it } from 'vitest'
import type { ChildProfile } from '../api'
import { checkpointLabel, currentAge, profileName } from './profile-formatters'

describe('profile formatters', () => {
  it('formats checkpoint identifiers for caregiver-facing copy', () => {
    expect(checkpointLabel('CP24M')).toBe('24-month check')
  })

  it('uses a privacy-preserving fallback when a display name is absent', () => {
    const profile = { display_name: '   ' } as ChildProfile
    expect(profileName(profile)).toBe('Your child')
  })

  it('calculates completed calendar months without producing negative ages', () => {
    const today = new Date('2026-10-29T00:00:00')
    expect(currentAge('2025-09-30', today)).toBe('1 year')
    expect(currentAge('2026-11-01', today)).toBe('0 months')
  })

  it('includes remaining months after a completed year', () => {
    expect(currentAge('2024-08-15', new Date('2026-10-29T00:00:00'))).toBe(
      '2 years, 2 months',
    )
  })
})

