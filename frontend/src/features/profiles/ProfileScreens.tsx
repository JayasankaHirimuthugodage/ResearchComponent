import { ArrowLeft, Baby, ChevronRight, Plus } from 'lucide-react'
import type { ChildProfile } from '../../api'
import { currentAge, profileName } from '../../app/profile-formatters'
import type { ProfileMode, View } from '../../app/types'

type ProfilesProps = {
  mode: ProfileMode
  profiles: ChildProfile[]
  loading: boolean
  error: string
  select: (profile: ChildProfile) => void
  showDetails: (profile: ChildProfile) => void
  setView: (view: View) => void
}

export function Profiles({ mode, profiles, loading, error, select, showDetails, setView }: ProfilesProps) {
  const selecting = mode === 'select-for-assessment'
  return <section className="page-content profile-page">
    {selecting ? <button className="back-link" onClick={() => setView('dashboard')} type="button"><ArrowLeft size={17} /> Mental development</button> : null}
    <p className="eyebrow">Child profiles</p>
    <h1>{selecting ? 'Choose a child for a development check' : 'Child profiles'}</h1>
    <p className="lead">{selecting ? 'Choose the child who will complete the development check.' : 'Add and view the child profiles saved to your caregiver account.'}</p>
    {error ? <p className="form-error">{error}</p> : null}
    <div className="profile-list">
      {loading ? <p>Loading profiles...</p> : profiles.map((profile) => selecting ? <button className="profile-card" key={profile.id} onClick={() => select(profile)} type="button"><span className="profile-card__icon"><Baby size={21} /></span><span><strong>{profileName(profile)}</strong><small>{currentAge(profile.date_of_birth)} old · Select for a development check</small></span><ChevronRight className="profile-card__arrow" size={20} /></button> : <article className="profile-card profile-card--static" key={profile.id}><span className="profile-card__icon"><Baby size={21} /></span><span><strong>{profileName(profile)}</strong><small>{currentAge(profile.date_of_birth)} old</small></span><button className="profile-card__details" onClick={() => showDetails(profile)} type="button">View details</button></article>)}
    </div>
    {!loading && profiles.length === 0 ? <p className="empty-state">No child profiles yet. Add one to begin.</p> : null}
    <button className="outline-button" onClick={() => setView('new-profile')} type="button"><Plus size={18} /> Add child profile</button>
  </section>
}

export function ProfileDetails({ profile, setView }: { profile: ChildProfile; setView: (view: View) => void }) {
  return <section className="page-content profile-page">
    <button className="back-link" onClick={() => setView('profiles')} type="button"><ArrowLeft size={17} /> Child profiles</button>
    <p className="eyebrow">Child profile</p><h1>{profileName(profile)}</h1>
    <p className="lead">Current age: <strong>{currentAge(profile.date_of_birth)}</strong></p>
    <section className="profile-details-card">
      <div><span>Date of birth</span><strong>{new Date(`${profile.date_of_birth}T00:00:00`).toLocaleDateString()}</strong></div>
      <div><span>Sex recorded at birth</span><strong>{profile.sex_recorded_at_birth === 'male' ? 'Male' : 'Female'}</strong></div>
      <div><span>Born prematurely</span><strong>{profile.premature_status === 'yes' ? 'Yes' : profile.premature_status === 'no' ? 'No' : 'Not sure'}</strong></div>
      {profile.premature_status === 'yes' && profile.gestational_age_weeks ? <div><span>Gestational age at birth</span><strong>{profile.gestational_age_weeks} weeks</strong></div> : null}
      <div><span>Birth-weight range</span><strong>{profile.birth_weight_range}</strong></div>
      <div><span>Languages spoken at home</span><strong>{profile.home_languages.join(', ')}</strong></div>
    </section>
  </section>
}

