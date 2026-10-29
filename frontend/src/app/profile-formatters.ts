import type { ChildProfile } from '../api'

export function checkpointLabel(id: string) {
  return id.replace(/^CP/, '').replace('M', '-month check')
}

export function profileName(profile: ChildProfile) {
  return profile.display_name?.trim() || 'Your child'
}

export function currentAge(dateOfBirth: string, today = new Date()) {
  const birth = new Date(`${dateOfBirth}T00:00:00`)
  let months =
    (today.getFullYear() - birth.getFullYear()) * 12 +
    today.getMonth() -
    birth.getMonth()

  if (today.getDate() < birth.getDate()) months -= 1
  months = Math.max(0, months)

  const years = Math.floor(months / 12)
  const remainingMonths = months % 12
  if (!years) {
    return `${remainingMonths} ${remainingMonths === 1 ? 'month' : 'months'}`
  }

  const yearLabel = `${years} ${years === 1 ? 'year' : 'years'}`
  if (!remainingMonths) return yearLabel

  return `${yearLabel}, ${remainingMonths} ${remainingMonths === 1 ? 'month' : 'months'}`
}

