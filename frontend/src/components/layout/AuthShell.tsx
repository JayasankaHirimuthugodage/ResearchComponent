import type { ReactNode } from 'react'

export function AuthShell({ children }: { children: ReactNode }) {
  return (
    <div className="auth-shell">
      <header className="auth-topbar">
        <strong><span>Child</span> Care</strong>
      </header>
      {children}
    </div>
  )
}

