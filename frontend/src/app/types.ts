export type View =
  | 'auth'
  | 'dashboard'
  | 'profiles'
  | 'profile-detail'
  | 'new-profile'
  | 'assessment-intro'
  | 'questionnaire'
  | 'results'
  | 'activity-suitability'
  | 'activity-plan'
  | 'activity-detail'

export type ProfileMode = 'manage' | 'select-for-assessment'

export type RecoverableView =
  | 'results'
  | 'activity-suitability'
  | 'activity-plan'
  | 'activity-detail'

export type DemoRecovery = {
  version: 1
  profile_id: string
  assessment_responses: Record<string, string>
  safety_responses?: Record<string, string>
  view: RecoverableView
  activity_title?: string
}

