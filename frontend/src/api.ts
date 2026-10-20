const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export type ChildProfile = { id: string; display_name: string | null; date_of_birth: string; sex_recorded_at_birth: 'male' | 'female'; premature_status: 'yes' | 'no' | 'not_sure'; gestational_age_weeks: number | null; birth_weight_range: string; home_languages: string[] }
export type Question = { id: string; domain: 'communication' | 'cognitive' | 'social_emotional'; text: string }
export type Questionnaire = { checkpoint_id: string; chronological_age_days: number; corrected_age_days: number | null; questions: Question[]; response_options: { code: string; label: string }[] }
export type Prediction = { status: 'complete' | 'incomplete'; checkpoint_id: string; message?: string; incomplete_question_ids?: string[]; overall_concern?: boolean; overall_message?: string; domains: { domain: string; title: string; decision: boolean; outcome: string; message: string }[] }
export type CaregiverExplanation = { status: 'complete' | 'incomplete'; checkpoint_id: string; notice?: string; message?: string; incomplete_question_ids?: string[]; domains: { domain: string; method: string; headline: string; observations: { question: string; questionnaire_domain: string; response: string; relationship: 'toward_further_attention' | 'away_from_further_attention' | 'response_variation' }[]; limitation: string }[] }
export type SafetyQuestion = { id: string; dimension: string; text: string }
export type ActivityResponse = { status: string; checkpoint_id: string; message: string; domains: { domain: string; status: string; activities: { title: string; instructions: string }[] }[] }
export type ProfileInput = Omit<ChildProfile, 'id'>

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) { super(message); this.status = status }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, { ...options, credentials: 'include', headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) } })
  if (response.status === 204) return undefined as T
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new ApiError(body.detail ?? 'The request could not be completed.', response.status)
  return body as T
}

export const api = {
  health: () => request<{ status: string }>('/health'),
  me: () => request<{ email: string; role: string }>('/api/v1/auth/me'),
  signUp: (email: string, password: string) => request<{ id: string; email: string; role: string }>('/api/v1/auth/sign-up', { method: 'POST', body: JSON.stringify({ email, password }) }),
  signIn: (email: string, password: string) => request<{ email: string; role: string }>('/api/v1/auth/sign-in', { method: 'POST', body: JSON.stringify({ email, password }) }),
  signOut: () => request<void>('/api/v1/auth/sign-out', { method: 'POST' }),
  profiles: () => request<ChildProfile[]>('/api/v1/child-profiles'),
  createProfile: (profile: ProfileInput) => request<ChildProfile>('/api/v1/child-profiles', { method: 'POST', body: JSON.stringify(profile) }),
  questionnaire: (profileId: string) => request<Questionnaire>(`/api/v1/child-profiles/${profileId}/questionnaire`),
  predict: (profileId: string, responses: Record<string, string>) => request<Prediction>(`/api/v1/child-profiles/${profileId}/prediction`, { method: 'POST', body: JSON.stringify({ responses }) }),
  explain: (profileId: string, responses: Record<string, string>) => request<CaregiverExplanation>(`/api/v1/child-profiles/${profileId}/explanation`, { method: 'POST', body: JSON.stringify({ responses }) }),
  safetyQuestions: () => request<{ schema_version: string; questions: SafetyQuestion[]; response_options: { code: string; label: string }[] }>('/api/v1/activity-suitability-questions'),
  activities: (profileId: string, responses: Record<string, string>, safety_responses: Record<string, string>) => request<ActivityResponse>(`/api/v1/child-profiles/${profileId}/activities`, { method: 'POST', body: JSON.stringify({ responses, safety_responses }) }),
}
