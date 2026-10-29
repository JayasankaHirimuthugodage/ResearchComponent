import { Menu } from 'lucide-react'
import type { ReactNode } from 'react'
import { Sidebar } from './Sidebar'

type AppShellProps = {
  activeLabel: string
  children: ReactNode
  email: string
  sidebarOpen: boolean
  closeSidebar: () => void
  navigate: (label: string) => void
  openSidebar: () => void
  signOut: () => void
}

export function AppShell({
  activeLabel,
  children,
  email,
  sidebarOpen,
  closeSidebar,
  navigate,
  openSidebar,
  signOut,
}: AppShellProps) {
  const title = activeLabel === 'Profile'
    ? 'Child Profiles'
    : 'Language, Cognitive and Socio-emotional Development'

  return (
    <div className="app-shell">
      <Sidebar
        activeLabel={activeLabel}
        email={email}
        onClose={closeSidebar}
        onNavigate={navigate}
        open={sidebarOpen}
        signOut={signOut}
      />
      {sidebarOpen ? (
        <button
          aria-label="Close navigation overlay"
          className="backdrop"
          onClick={closeSidebar}
          type="button"
        />
      ) : null}
      <main className="main-content">
        <header className="topbar">
          <button
            aria-label="Open navigation"
            className="icon-button"
            onClick={openSidebar}
            type="button"
          >
            <Menu size={23} />
          </button>
          <span className="topbar__title">{title}</span>
          <span className="avatar avatar--top">{email.slice(0, 1).toUpperCase()}</span>
        </header>
        {children}
      </main>
    </div>
  )
}

