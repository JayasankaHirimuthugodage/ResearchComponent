import { useState } from 'react'
import type { Questionnaire } from '../../api'
import { DOMAIN_LABELS } from '../../app/constants'

export function QuestionnaireScreen({ questionnaire, complete }: { questionnaire: Questionnaire; complete: (responses: Record<string, string>) => Promise<void> }) {
  const [index, setIndex] = useState(0)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const question = questionnaire.questions[index]
  const answer = answers[question.id]
  const last = index === questionnaire.questions.length - 1
  const progress = ((index + 1) / questionnaire.questions.length) * 100

  async function next() {
    if (!answer) return
    if (!last) return setIndex((value) => value + 1)
    setBusy(true)
    setError('')
    try { await complete(answers) } catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to process the screening.') } finally { setBusy(false) }
  }

  const domainClass = question.domain === 'cognitive' ? 'question-domain--pink' : question.domain === 'social_emotional' ? 'question-domain--neutral' : ''
  return <section className="page-content questionnaire-page">
    <div className="questionnaire-progress"><div className="question-meta"><span>Question {index + 1} of {questionnaire.questions.length}</span><span>{Math.round(progress)}% complete</span></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div></div>
    <div className="questionnaire-workspace">
      <div className="questionnaire-prompt"><span className={`question-domain ${domainClass}`}>{DOMAIN_LABELS[question.domain]}</span><h1>{question.text}</h1><p className="lead">Choose the answer that best represents what you usually observe.</p></div>
      <div className="questionnaire-response"><section className="question-card"><div className="answer-list">{questionnaire.response_options.map((option) => <label className={`answer-option ${answer === option.code ? 'answer-option--selected' : ''}`} key={option.code}><input checked={answer === option.code} name="answer" onChange={() => setAnswers((all) => ({ ...all, [question.id]: option.code }))} type="radio" /><span>{option.label}</span></label>)}</div></section>{error ? <p className="form-error">{error}</p> : null}<div className="question-actions"><button className="outline-button" disabled={index === 0 || busy} onClick={() => setIndex((value) => value - 1)} type="button">Previous</button><button className="primary-button primary-button--active" disabled={!answer || busy} onClick={next} type="button">{busy ? 'Preparing prediction...' : last ? 'View prediction' : 'Next question'}</button></div></div>
    </div>
    <p className="question-privacy-note">Your answers will be used to prepare the three-month monitoring prediction.</p>
  </section>
}

