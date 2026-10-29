import { ArrowLeft } from 'lucide-react'
import type { ChildProfile, Questionnaire } from '../../api'
import { checkpointLabel, profileName } from '../../app/profile-formatters'
import type { View } from '../../app/types'

export function AssessmentIntro({ profile, questionnaire, setView }: { profile: ChildProfile; questionnaire: Questionnaire; setView: (view: View) => void }) {
  return <section className="page-content assessment-intro">
    <button className="back-link" onClick={() => setView('dashboard')} type="button"><ArrowLeft size={17} /> Back to dashboard</button>
    <div className="assessment-intro__lead">
      <p className="eyebrow">{checkpointLabel(questionnaire.checkpoint_id)}</p>
      <h1>Before you begin</h1>
      <p className="lead">This check asks about everyday observations across three development areas. Answer based on what you usually see.</p>
      <button className="primary-button primary-button--active" onClick={() => setView('questionnaire')} type="button">Begin questions</button>
    </div>
    <div className="assessment-intro__details">
      <section className="assessment-summary"><div className="assessment-count">{questionnaire.questions.length}</div><div><h2>Questions for {profileName(profile)}</h2><p>This development check is matched to your child&apos;s current age.</p></div></section>
      <div className="area-list"><div><span className="area-dot area-dot--blue" /><div><strong>Communication</strong><small>Everyday words and understanding</small></div><span className="included-pill">Included</span></div><div><span className="area-dot area-dot--pink" /><div><strong>Cognitive</strong><small>Thinking and problem-solving</small></div><span className="included-pill">Included</span></div><div><span className="area-dot" /><div><strong>Social-emotional</strong><small>Feelings and social responses</small></div><span className="included-pill">Included</span></div></div>
      <section className="notice-card"><strong>This is a screening support tool.</strong><p>It does not diagnose a developmental condition or replace professional assessment.</p></section>
    </div>
  </section>
}

