import { Link, useLocation } from 'react-router-dom';
import sidebarLogo from '../assets/sidebar-logo.png';

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="1.8" />
      <path d="M21 21L16.5 16.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function ClockIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 7V12L15.5 14.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Sidebar() {
  const location = useLocation();

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <img src={sidebarLogo} alt="Satyata logo" className="sidebar-logo-img" />
      </div>

      <p className="sidebar-tagline">
        Bilingual Nepali-English Fake News Detection
      </p>

      <nav className="sidebar-nav">
        <Link to="/" className={`nav-item ${location.pathname === '/' ? 'active' : ''}`}>
          <SearchIcon />
          <span>Verify News</span>
        </Link>

        <Link to="/history" className={`nav-item ${location.pathname === '/history' ? 'active' : ''}`}>
          <ClockIcon />
          <span>History</span>
        </Link>
      </nav>
    </aside>
  );
}

export default Sidebar;