import {
  Activity,
  Brain,
  ChartNoAxesColumnIncreasing,
  ChevronDown,
  Syringe,
  UserRound,
  X,
} from 'lucide-react'

type SidebarProps = {
  activeLabel: string
  open: boolean
  onClose: () => void
  email: string
  onNavigate: (label: string) => void
  signOut: () => void
}

const navigation = [
  ['Profile', UserRound],
  ['Child Growth', ChartNoAxesColumnIncreasing],
  ['Vaccination', Syringe],
  ['ECC Screening', Activity],
  ['Mental Development', Brain],
] as const

export function Sidebar({
  activeLabel,
  open,
  onClose,
  email,
  onNavigate,
  signOut,
}: SidebarProps) {
  return (
    <aside
      aria-label="Primary navigation"
      className={`sidebar ${open ? 'sidebar--open' : ''}`}
    >
      <div className="sidebar__brand">
        <span>Child</span> Care
        <button
          aria-label="Close navigation"
          className="icon-button sidebar__close"
          onClick={onClose}
          type="button"
        >
          <X size={20} />
        </button>
      </div>
      <nav className="sidebar__nav">
        {navigation.map(([label, Icon]) => (
          <button
            className={`nav-item ${label === activeLabel ? 'nav-item--active' : ''}`}
            key={label}
            onClick={() => onNavigate(label)}
            type="button"
          >
            <Icon size={18} />
            <span>{label}</span>
            {label !== 'Profile' ? <ChevronDown className="nav-item__chevron" size={16} /> : null}
          </button>
        ))}
      </nav>
      <div className="sidebar__footer">
        <span className="avatar">{email.slice(0, 1).toUpperCase()}</span>
        <div>
          <strong>Caregiver account</strong>
          <small>{email}</small>
          <button className="signout-button" onClick={signOut} type="button">Sign out</button>
        </div>
      </div>
    </aside>
  )
}

