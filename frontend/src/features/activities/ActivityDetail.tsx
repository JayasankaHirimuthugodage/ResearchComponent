import { ArrowLeft } from 'lucide-react'
import type { View } from '../../app/types'

type ActivityDetailProps = {
  detail: { title: string; instructions: string }
  setView: (view: View) => void
}

export function ActivityDetail({ detail, setView }: ActivityDetailProps) {
  return (
    <section className="page-content activity-page">
      <button className="back-link" onClick={() => setView('activity-plan')} type="button">
        <ArrowLeft size={17} /> Activity plan
      </button>
      <p className="eyebrow">Activity guidance</p>
      <h1>{detail.title}</h1>
      <section className="activity-detail-card">
        <h2>How to use this activity</h2>
        <p>{detail.instructions}</p>
      </section>
      <section className="notice-card">
        <strong>Use safe, age-appropriate objects and supervision.</strong>
        <p>
          Stop if your child becomes uncomfortable, and seek professional guidance when
          appropriate.
        </p>
      </section>
      <button
        className="primary-button primary-button--active"
        onClick={() => setView('activity-plan')}
        type="button"
      >
        Back to activities
      </button>
    </section>
  )
}

