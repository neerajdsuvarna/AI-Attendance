import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import './Dashboard.css'

function Dashboard({ user, userRole }) {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)

  const handleLogout = async () => {
    setLoading(true)
    await supabase.auth.signOut()
    navigate('/login')
  }

  const getRoleDisplay = (role) => {
    const roleMap = {
      'super_admin': 'Super Administrator',
      'hr_admin': 'HR Administrator',
      'employee': 'Employee'
    }
    return roleMap[role] || role
  }

  // Check if user has admin privileges
  const isAdmin = userRole === 'super_admin' || userRole === 'hr_admin'
  const isSuperAdmin = userRole === 'super_admin'

  const getRoleColor = (role) => {
    const colorMap = {
      'super_admin': '#dc2626',
      'hr_admin': '#2563eb',
      'employee': '#059669'
    }
    return colorMap[role] || '#666'
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div>
          <h1>Dashboard</h1>
          <p className="user-email">{user?.email}</p>
        </div>
        <div className="header-right">
          <span 
            className="role-badge"
            style={{ 
              background: `linear-gradient(135deg, ${getRoleColor(userRole)} 0%, ${getRoleColor(userRole)}dd 100%)`,
              boxShadow: `0 4px 12px ${getRoleColor(userRole)}40`
            }}
          >
            {getRoleDisplay(userRole)}
          </span>
          <button 
            onClick={handleLogout} 
            className="logout-button"
            disabled={loading}
          >
            {loading ? 'Logging out...' : 'Logout'}
          </button>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="welcome-card">
          <h2>Welcome back! 👋</h2>
          <p>You are logged in as <strong>{getRoleDisplay(userRole)}</strong></p>
          {isSuperAdmin && (
            <p style={{ marginTop: '0.5rem', fontSize: '0.95rem', opacity: 0.9 }}>
              ⚡ You have full system access with Super Administrator privileges
            </p>
          )}
          {userRole === 'hr_admin' && (
            <p style={{ marginTop: '0.5rem', fontSize: '0.95rem', opacity: 0.9 }}>
              👔 You have HR Administrator access to manage employees and attendance
            </p>
          )}
          {userRole === 'employee' && (
            <p style={{ marginTop: '0.5rem', fontSize: '0.95rem', opacity: 0.9 }}>
              👤 You have employee access to view your attendance records
            </p>
          )}
        </div>

        <div className="dashboard-grid">
          {(isSuperAdmin || userRole === 'hr_admin') && (
            <div 
              className="dashboard-card"
              onClick={() => navigate('/register-employee')}
              style={{ cursor: 'pointer' }}
            >
              <h3>Register Employee</h3>
              <p>
                {isSuperAdmin
                  ? 'Add new employees to the system with face recognition registration'
                  : 'Add new employees to the system (Employee role only)'
                }
              </p>
            </div>
          )}
          {(isAdmin || userRole === 'employee') && (
            <div 
              className="dashboard-card"
              onClick={() => navigate('/employees')}
              style={{ cursor: 'pointer' }}
            >
              <h3>Employees</h3>
              <p>
                {isSuperAdmin 
                  ? 'Full access: Manage all employee records, add new team members, and update profiles'
                  : userRole === 'hr_admin'
                  ? 'Manage employee records, add new team members, and update profiles'
                  : 'View your employee profile and information'
                }
              </p>
            </div>
          )}
          {(isAdmin || userRole === 'employee') && (
            <div 
              className="dashboard-card"
              onClick={() => navigate('/attendance')}
              style={{ cursor: 'pointer' }}
            >
              <h3>Attendance</h3>
              <p>
                {isSuperAdmin
                  ? 'Full access: View and manage all attendance records, track all check-ins and check-outs'
                  : userRole === 'hr_admin'
                  ? 'View and manage attendance records, track check-ins and check-outs'
                  : 'View your personal attendance records and check-in/check-out history'
                }
              </p>
            </div>
          )}
          {(isSuperAdmin || userRole === 'hr_admin') && (
            <div 
              className="dashboard-card live-detection-card"
              onClick={() => navigate('/live-detection')}
              style={{ cursor: 'pointer' }}
            >
              <h3>Live Detection</h3>
              <p>
                {isSuperAdmin
                  ? 'Real-time face recognition for attendance tracking. Monitor live check-ins and check-outs'
                  : 'Real-time face recognition for attendance tracking. Monitor employee check-ins and check-outs'
                }
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard