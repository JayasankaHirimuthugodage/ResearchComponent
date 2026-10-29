import { ArrowLeft, Info } from 'lucide-react'
import { useState } from 'react'
import type { ActivityResponse, SafetyQuestion } from '../../api'
import type { View } from '../../app/types'
import { buildActivityCards } from './activity-cards'

export function ActivitySuitability({ questions, options, submit, setView }: { questions: SafetyQuestion[]; options: { code: string; label: string }[]; submit: (responses: Record<string, string>) => Promise<void>; setView: (view: View) => void }) {
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const answeredCount = questions.filter((question) => Boolean(answers[question.id])).length
  const complete = answeredCount === questions.length

  async function send() {
    if (Object.keys(answers).length !== questions.length) return setError('Please answer every activity-suitability question.')
    setBusy(true); setError('')
    try { await submit(answers) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to load activities.') } finally { setBusy(false) }
  }

  return <section className="page-content activity-page">
    <button className="back-link" onClick={() => setView('results')} type="button"><ArrowLeft size={17} /> Prediction results</button>
    <p className="eyebrow">Activity support</p><h1>A few questions about current abilities</h1>
    <p className="lead">These answers help identify age-appropriate activity suggestions.</p>
    <div className="activity-form-progress"><div><strong>{complete ? 'All questions answered' : `${answeredCount} of ${questions.length} answered`}</strong><span>{complete ? 'Ready to find suitable activities.' : 'Please answer each question.'}</span></div><div aria-label={`${answeredCount} of ${questions.length} questions answered`} aria-valuemax={questions.length} aria-valuemin={0} aria-valuenow={answeredCount} className="activity-progress-track" role="progressbar"><span style={{ width: `${questions.length ? (answeredCount / questions.length) * 100 : 0}%` }} /></div></div>
    <div className="suitability-list">{questions.map((question) => {
      const heading = question.dimension.replaceAll('_', ' ').replace(/^./, (character) => character.toUpperCase())
      return <fieldset className={`suitability-card ${answers[question.id] ? 'suitability-card--answered' : ''}`} key={question.id}><legend className="sr-only">{heading}</legend><div className="suitability-card__heading"><span aria-hidden="true" className="suitability-card__marker" /><h2 aria-hidden="true">{heading}</h2></div><p>{question.text}</p><div className="suitability-options">{options.map((option) => <label className={answers[question.id] === option.code ? 'suitability-option suitability-option--selected' : 'suitability-option'} key={option.code}><input checked={answers[question.id] === option.code} name={question.id} onChange={() => { setAnswers((all) => ({ ...all, [question.id]: option.code })); setError('') }} type="radio" value={option.code} /><span>{option.label}</span></label>)}</div></fieldset>
    })}</div>
    {error ? <p className="form-error">{error}</p> : null}
    <div className="activity-form-footer"><section className="notice-card"><strong>Activity suggestions support everyday development.</strong><p>They do not diagnose a condition or replace professional advice.</p></section><button className="primary-button primary-button--active" disabled={!complete || busy} onClick={send} type="button">{busy ? 'Finding activities...' : 'See personalised activities'}</button></div>
  </section>
}

export function ActivityPlan({ activities, select, setView }: { activities: ActivityResponse; select: (title: string, instructions: string) => void; setView: (view: View) => void }) {
  const cards = buildActivityCards(activities)
  return <section className="page-content activity-page">
    <button className="back-link" onClick={() => setView('results')} type="button"><ArrowLeft size={17} /> Prediction results</button>
    <p className="eyebrow">Activity plan</p><h1>Activities for your child</h1><p className="lead">{activities.message}</p>
    {cards.length ? <div className="activity-card-list">{cards.map((activity, index) => <article className={`activity-card ${index % 2 ? 'activity-card--pink' : ''}`} key={`${activity.title}-${activity.instructions}`}><span className="activity-number">{index + 1}</span><div className="activity-card__content"><span className="activity-domain">{activity.domains.join(' · ')}</span><h2>{activity.title}</h2><p>Open the activity to see the steps and safety guidance.</p></div><button aria-label={`View activity: ${activity.title}`} onClick={() => select(activity.title, activity.instructions)} type="button">View activity</button></article>)}</div> : <section className="results-placeholder"><Info size={21} /><div><strong>No targeted activity cards are available.</strong><p>{activities.message}</p></div></section>}
    <button className="primary-button primary-button--active" onClick={() => setView('dashboard')} type="button">Return to dashboard</button>
  </section>
}

