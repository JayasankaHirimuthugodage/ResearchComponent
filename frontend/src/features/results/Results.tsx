import { ArrowLeft, ChevronDown, Flag, Info } from 'lucide-react'
import type { CaregiverExplanation, Prediction } from '../../api'
import { DOMAIN_LABELS } from '../../app/constants'
import type { View } from '../../app/types'

type ExplanationObservation = CaregiverExplanation['domains'][number]['observations'][number]

function ExplanationObservationItem({ observation, sourceAreaMessage }: { observation: ExplanationObservation; sourceAreaMessage?: string }) {
  const sourceArea = DOMAIN_LABELS[observation.questionnaire_domain] ?? observation.questionnaire_domain.replaceAll('_', ' ')
  return <li><span className="observation-domain">From the {sourceArea.toLowerCase()} questions</span><p>{observation.question}</p><span className="observation-response">Your answer: <strong>{observation.response}</strong></span>{sourceAreaMessage ? <small className="observation-area-status">{sourceAreaMessage}</small> : null}</li>
}

export function Results({ result, explanation, explanationLoading, explanationError, goActivities, setView }: { result: Prediction; explanation: CaregiverExplanation | null; explanationLoading: boolean; explanationError: string; goActivities: () => void; setView: (view: View) => void }) {
  if (result.status === 'incomplete') return <section className="page-content results-page"><h1>More answers are needed</h1><p className="lead">{result.message}</p><button className="primary-button primary-button--active" onClick={() => setView('questionnaire')} type="button">Return to questions</button></section>
  const byDomain = new Map(explanation?.domains.map((item) => [item.domain, item]))
  const flaggedDomains = result.domains.filter((domain) => domain.decision)
  const unflaggedDomains = result.domains.filter((domain) => !domain.decision)
  return <section className="page-content results-page">
    <button className="back-link" onClick={() => setView('dashboard')} type="button"><ArrowLeft size={17} /> Back to dashboard</button>
    <p className="eyebrow">Research prediction</p><h1>Three-month developmental monitoring prediction</h1><p className="lead">{result.overall_message}</p>
    {flaggedDomains.length ? <><section className="results-attention-heading"><span className="results-attention-heading__icon"><Flag aria-hidden="true" size={21} /></span><div><h2>Areas that may benefit from further attention</h2><p>These results suggest continued observation and, when appropriate, a conversation with a qualified healthcare professional.</p></div><span className="results-attention-heading__count">{flaggedDomains.length} of {result.domains.length} {result.domains.length === 1 ? 'area' : 'areas'}</span></section><div className="result-list">
      {flaggedDomains.map((domain) => {
        const details = byDomain.get(domain.domain)
        const sameAreaObservations = details?.observations.filter((observation) => observation.questionnaire_domain === domain.domain) ?? []
        const otherAreaObservations = details?.observations.filter((observation) => observation.questionnaire_domain !== domain.domain) ?? []
        return <section className={`result-card result-card--detailed ${domain.domain === 'cognitive' ? 'result-card--pink' : domain.domain === 'social_emotional' ? 'result-card--neutral' : ''}`} key={domain.domain}><div className="result-card__content"><div className="result-card__heading"><span className="result-card__flag"><Flag aria-hidden="true" size={17} /></span><h2>{domain.title}</h2><span className="result-card__status">Further attention suggested</span></div><p className="result-card__message">Additional monitoring may be helpful.</p>
          {details ? <details className="caregiver-explanation"><summary>Why this result?<ChevronDown aria-hidden="true" size={17} /></summary><div className="caregiver-explanation__content"><p className="explanation-intro">All your answers were considered together. The answers below played the largest part in this result, but no single answer decided it.</p>{sameAreaObservations.length ? <section className="observation-group"><h3>Answers from this development area that mattered most</h3><ul className="explanation-observations">{sameAreaObservations.map((observation) => <ExplanationObservationItem key={observation.question} observation={observation} />)}</ul></section> : <p className="explanation-empty">No single answer from this development area stood out on its own.</p>}
          {otherAreaObservations.length ? <details className="observation-more observation-more--other"><summary>Other answers that also played a part <span>{otherAreaObservations.length}</span></summary><div className="observation-more__content"><p>Some answers from other parts of the check also played a part. Children&apos;s skills can be connected across development areas.</p><ul className="explanation-observations">{otherAreaObservations.map((observation) => { const sourceResult = result.domains.find((item) => item.domain === observation.questionnaire_domain); const sourceName = DOMAIN_LABELS[observation.questionnaire_domain] ?? observation.questionnaire_domain.replaceAll('_', ' '); const sourceAreaMessage = sourceResult?.decision ? `${sourceName} is shown separately in your results.` : `${sourceName} was not separately highlighted for further monitoring.`; return <ExplanationObservationItem key={observation.question} observation={observation} sourceAreaMessage={sourceAreaMessage} /> })}</ul></div></details> : null}<p className="explanation-limitation">These answers do not identify a cause or provide a diagnosis.</p></div></details> : null}</div></section>
      })}</div>{unflaggedDomains.length ? <p className="unflagged-summary">The other development {unflaggedDomains.length === 1 ? 'area was' : 'areas were'} not identified for additional monitoring by this prediction.</p> : null}</> : <section className="results-placeholder results-placeholder--clear"><Info size={22} /><div><strong>No development areas were identified for additional monitoring by this three-month prediction.</strong><p>This does not rule out developmental needs.</p></div></section>}
    {flaggedDomains.length && explanationLoading ? <p className="explanation-loading">Preparing caregiver-friendly explanations...</p> : null}
    {flaggedDomains.length && explanationError ? <p className="form-error">Screening results are available, but the explanation could not be prepared: {explanationError}</p> : null}
    {flaggedDomains.length && explanation?.notice ? <details className="explanation-notice"><summary>About this explanation</summary><p>{explanation.notice}</p></details> : null}
    <section className="notice-card"><strong>This research prediction is not a diagnosis or a guarantee of future development.</strong><p>Speak with a qualified healthcare professional if you have questions or concerns about your child&apos;s development.</p></section>
    <div className="result-actions">{flaggedDomains.length ? <button className="primary-button primary-button--active" onClick={goActivities} type="button">Explore activity support</button> : null}<button className="text-button" onClick={() => setView('dashboard')} type="button">Return to dashboard</button></div>
  </section>
}

