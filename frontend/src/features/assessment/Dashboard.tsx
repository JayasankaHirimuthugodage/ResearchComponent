import { ClipboardList, Sparkles } from 'lucide-react'
import type { ChildProfile, Questionnaire } from '../../api'
import { checkpointLabel, profileName } from '../../app/profile-formatters'
import type { View } from '../../app/types'

type DashboardProps = {
  profile: ChildProfile | null
  questionnaire: Questionnaire | null
  startSelection: () => void
  setView: (view: View) => void
}

export function Dashboard({
  profile,
  questionnaire,
  startSelection,
  setView,
}: DashboardProps) {
  if (!profile || !questionnaire) {
    return (
      <section className="page-content">
        <h1>Ready to begin a development check?</h1>
        <p className="lead">
          Choose a child profile to see the development check prepared for their age.
        </p>
        <section className="next-card">
          <h2>Select child profile</h2>
          <p>Choose the child who will complete the development check.</p>
          <button
            className="primary-button primary-button--active"
            onClick={startSelection}
            type="button"
          >
            Select child profile
          </button>
        </section>
      </section>
    )
  }

  return (
    <section className="page-content dashboard-page">
      <h1>{profileName(profile)}&apos;s developmental check</h1>
      <p className="lead">
        {checkpointLabel(questionnaire.checkpoint_id)} &middot; An age-appropriate caregiver
        observation check is ready.
      </p>
      <section className="ready-card">
        <div className="ready-card__icon"><Sparkles size={22} /></div>
        <div>
          <p className="ready-card__eyebrow">Age-appropriate check</p>
          <h2>{checkpointLabel(questionnaire.checkpoint_id)}</h2>
          <p>
            {questionnaire.questions.length} questions across communication, cognitive,
            and social-emotional development.
          </p>
        </div>
        <span className="status-pill">Ready now</span>
      </section>
      <h2 className="section-heading">Development areas</h2>
      <div className="domain-chips">
        <span className="chip chip--blue">Communication</span>
        <span className="chip chip--pink">Cognitive</span>
        <span className="chip">Social-emotional</span>
      </div>
      <section className="previous-card">
        <span className="previous-card__icon"><ClipboardList size={20} /></span>
        <div>
          <h2>Before you begin</h2>
          <p>
            Choose the answer that best represents what you usually observe. This screening
            supports early monitoring and does not provide a diagnosis.
          </p>
        </div>
      </section>
      <button
        className="primary-button primary-button--active"
        onClick={() => setView('assessment-intro')}
        type="button"
      >
        Start assessment
      </button>
      <button className="text-button" onClick={startSelection} type="button">
        Choose another child
      </button>
    </section>
  )
}

