import { ArrowLeft } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { api, type ChildProfile, type ProfileInput } from '../../api'
import { HOME_LANGUAGES } from '../../app/constants'
import type { View } from '../../app/types'

export function NewProfile({ created, setView }: { created: (profile: ChildProfile) => void; setView: (view: View) => void }) {
  const [premature, setPremature] = useState<'yes' | 'no' | 'not_sure' | ''>('')
  const [languagesSelected, setLanguagesSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  function toggleLanguage(value: string) {
    setLanguagesSelected((items) => items.includes(value) ? items.filter((item) => item !== value) : [...items, value])
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError('')
    if (!languagesSelected.length) return setError('Select at least one home language.')
    const fields = new FormData(event.currentTarget)
    const weeks = fields.get('gestational_age_weeks')
    const payload: ProfileInput = {
      display_name: String(fields.get('display_name') || '') || null,
      date_of_birth: String(fields.get('date_of_birth')),
      sex_recorded_at_birth: String(fields.get('sex_recorded_at_birth')) as 'male' | 'female',
      premature_status: premature as 'yes' | 'no' | 'not_sure',
      gestational_age_weeks: premature === 'yes' && weeks ? Number(weeks) : null,
      birth_weight_range: String(fields.get('birth_weight_range')),
      home_languages: languagesSelected,
    }
    setBusy(true)
    try { created(await api.createProfile(payload)) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to save the child profile.') } finally { setBusy(false) }
  }

  return <section className="page-content profile-page">
    <button className="back-link" onClick={() => setView('profiles')} type="button"><ArrowLeft size={17} /> Child profiles</button>
    <p className="eyebrow">New child profile</p><h1>Tell us about your child</h1>
    <form className="profile-form" onSubmit={submit}>
      <fieldset><legend>Child details</legend><label><span>Child&apos;s name <em>optional</em></span><input name="display_name" placeholder="Name used in this app" /></label><label><span>Date of birth</span><input max={new Date().toISOString().slice(0, 10)} name="date_of_birth" required type="date" /></label><div className="form-group"><span>Sex recorded at birth</span><div className="radio-row"><label><input name="sex_recorded_at_birth" required type="radio" value="male" /> Male</label><label><input name="sex_recorded_at_birth" type="radio" value="female" /> Female</label></div></div></fieldset>
      <fieldset><legend>Birth information</legend><div className="form-group"><span>Was your child born before 37 completed weeks?</span><div className="radio-row radio-row--stack">{(['yes', 'no', 'not_sure'] as const).map((value) => <label key={value}><input checked={premature === value} name="premature" onChange={() => setPremature(value)} required type="radio" /> {value === 'not_sure' ? 'Not sure' : value[0].toUpperCase() + value.slice(1)}</label>)}</div></div>{premature === 'yes' ? <label><span>Gestational age at birth</span><select defaultValue="" name="gestational_age_weeks" required><option disabled value="">Select completed weeks</option>{Array.from({ length: 17 }, (_, index) => index + 20).map((weeks) => <option key={weeks} value={weeks}>{weeks} weeks</option>)}</select></label> : null}<label><span>Birth-weight range</span><select defaultValue="" name="birth_weight_range" required><option disabled value="">Select a range</option><option>Less than 1.5 kg</option><option>1.5–1.99 kg</option><option>2.0–2.49 kg</option><option>2.5–2.99 kg</option><option>3.0–3.49 kg</option><option>3.5–3.99 kg</option><option>4.0 kg or more</option><option>Not sure / Record unavailable</option></select></label></fieldset>
      <fieldset><legend>Languages spoken to your child</legend><p className="field-description">Select one or more languages regularly spoken by parents or caregivers.</p><div className="language-grid">{HOME_LANGUAGES.map((value) => <label key={value}><input checked={languagesSelected.includes(value)} onChange={() => toggleLanguage(value)} type="checkbox" /> {value}</label>)}</div></fieldset>
      {error ? <p className="form-error">{error}</p> : null}<button className="primary-button primary-button--active" disabled={busy} type="submit">{busy ? 'Saving...' : 'Save child profile'}</button>
    </form>
  </section>
}

