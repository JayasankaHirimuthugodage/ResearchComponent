import { useState, type FormEvent } from 'react'
import { api } from '../../api'

type AuthMode = 'sign-in' | 'sign-up'

export function AuthScreen({ onAuthenticated }: { onAuthenticated: (email: string) => void }) {
  const [mode, setMode] = useState<AuthMode>('sign-in')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    if (mode === 'sign-up' && password !== confirm) {
      setError('The two passwords do not match.')
      return
    }
    setBusy(true)
    try {
      if (mode === 'sign-up') await api.signUp(email, password)
      const session = await api.signIn(email, password)
      onAuthenticated(session.email)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Unable to continue.')
    } finally {
      setBusy(false)
    }
  }

  function switchMode() {
    setMode(mode === 'sign-in' ? 'sign-up' : 'sign-in')
    setError('')
  }

  return (
    <section className="auth-layout" aria-labelledby="auth-title">
      <div className="auth-intro">
        <p className="eyebrow">AI-Driven Multi-Model System for Child Health</p>
        <h1 id="auth-title">Helping you support your child&apos;s healthy development.</h1>
        <p className="lead">Sign in or create an account to continue.</p>
      </div>
      <section className="auth-card">
        <div className="auth-card__heading">
          <h2>{mode === 'sign-in' ? 'Welcome back' : 'Create caregiver account'}</h2>
          <p>{mode === 'sign-in' ? 'Sign in to continue with your child profiles.' : 'Use an email and a password of at least 12 characters.'}</p>
        </div>
        <form onSubmit={submit}>
          {mode === 'sign-up' ? <label><span>Your name <em>optional</em></span><input placeholder="Caregiver name" /></label> : null}
          <label><span>Email address</span><input autoComplete="email" onChange={(event) => setEmail(event.target.value)} required type="email" value={email} /></label>
          <label><span>Password</span><input autoComplete={mode === 'sign-in' ? 'current-password' : 'new-password'} minLength={12} onChange={(event) => setPassword(event.target.value)} required type="password" value={password} /></label>
          {mode === 'sign-up' ? <label><span>Confirm password</span><input autoComplete="new-password" minLength={12} onChange={(event) => setConfirm(event.target.value)} required type="password" value={confirm} /></label> : null}
          {error ? <p className="form-error">{error}</p> : null}
          <button className="primary-button primary-button--full" disabled={busy} type="submit">{busy ? 'Please wait...' : mode === 'sign-in' ? 'Sign in' : 'Create account'}</button>
        </form>
        <div className="auth-card__switch">
          <span>{mode === 'sign-in' ? 'New caregiver?' : 'Already have an account?'}</span>
          <button onClick={switchMode} type="button">{mode === 'sign-in' ? 'Create an account' : 'Sign in instead'}</button>
        </div>
      </section>
    </section>
  )
}

