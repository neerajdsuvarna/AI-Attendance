import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import './Employees.css'

function Employees({ userRole }) {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [hrAdmins, setHrAdmins] = useState([])
  const [employees, setEmployees] = useState([])
  const [activeTab, setActiveTab] = useState('all') // 'all', 'hr_admins', 'employees'

  useEffect(() => {
    fetchEmployees()
  }, [])

  const fetchEmployees = async () => {
    try {
      setLoading(true)
      setError(null)

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        setError('Not authenticated. Please log in again.')
        return
      }

      const { data, error: fetchError } = await supabase.functions.invoke('get-employees-list', {
        headers: {
          Authorization: `Bearer ${session.access_token}`,
        },
      })

      if (fetchError) throw fetchError

      if (data.error) {
        throw new Error(data.error)
      }

      if (data.success) {
        setHrAdmins(data.hr_admins || [])
        setEmployees(data.employees || [])
      } else {
        throw new Error('Failed to fetch employees')
      }
    } catch (err) {
      console.error('Error fetching employees:', err)
      setError(err.message || 'Failed to load employees. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const getRoleBadgeColor = (role) => {
    switch (role) {
      case 'super_admin':
        return '#dc2626'
      case 'hr_admin':
        return '#2563eb'
      case 'employee':
        return '#059669'
      default:
        // Default to employee color for null/undefined roles
        return '#059669'
    }
  }

  const getRoleDisplay = (role) => {
    switch (role) {
      case 'super_admin':
        return 'Super Admin'
      case 'hr_admin':
        return 'HR Admin'
      case 'employee':
        return 'Employee'
      default:
        // If role is null, undefined, or any other value, display as Employee
        return 'Employee'
    }
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  const filteredList = () => {
    switch (activeTab) {
      case 'hr_admins':
        return hrAdmins
      case 'employees':
        return employees
      case 'all':
      default:
        return [...hrAdmins, ...employees]
    }
  }

  return (
    <div className="employees-container">
      <div className="employees-header">
        <button
          onClick={() => navigate('/dashboard')}
          className="back-button"
        >
          ← Back to Dashboard
        </button>
        <h1>Employees</h1>
        <p>View and manage all employees and HR administrators</p>
      </div>

      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={fetchEmployees} className="retry-button">
            Retry
          </button>
        </div>
      )}

      {loading ? (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading employees...</p>
        </div>
      ) : (
        <>
          <div className="employees-tabs">
            <button
              className={`tab-button ${activeTab === 'all' ? 'active' : ''}`}
              onClick={() => setActiveTab('all')}
            >
              All ({hrAdmins.length + employees.length})
            </button>
            <button
              className={`tab-button ${activeTab === 'hr_admins' ? 'active' : ''}`}
              onClick={() => setActiveTab('hr_admins')}
            >
              HR Admins ({hrAdmins.length})
            </button>
            <button
              className={`tab-button ${activeTab === 'employees' ? 'active' : ''}`}
              onClick={() => setActiveTab('employees')}
            >
              Employees ({employees.length})
            </button>
          </div>

          <div className="employees-list">
            {filteredList().length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">👥</div>
                <h3>No employees found</h3>
                <p>
                  {activeTab === 'all'
                    ? 'No employees or HR admins registered yet.'
                    : activeTab === 'hr_admins'
                    ? 'No HR administrators registered yet.'
                    : 'No employees registered yet.'}
                </p>
                {(userRole === 'super_admin' || userRole === 'hr_admin') && (
                  <button
                    onClick={() => navigate('/register-employee')}
                    className="register-button"
                  >
                    Register New Employee
                  </button>
                )}
              </div>
            ) : (
              <div className="employees-grid">
                {filteredList().map((person) => (
                  <div key={person.id} className="employee-card">
                    <div className="employee-header">
                      <div className="employee-avatar">
                        {person.name.charAt(0).toUpperCase()}
                      </div>
                      <div className="employee-info">
                        <h3 className="employee-name">{person.name}</h3>
                        <p className="employee-email">{person.email}</p>
                      </div>
                    </div>
                    <div className="employee-details">
                      <div className="detail-item">
                        <span className="detail-label">Role:</span>
                        <span
                          className="role-badge"
                          style={{
                            background: `linear-gradient(135deg, ${getRoleBadgeColor(person.role)} 0%, ${getRoleBadgeColor(person.role)}dd 100%)`,
                            boxShadow: `0 4px 12px ${getRoleBadgeColor(person.role)}40`
                          }}
                        >
                          {getRoleDisplay(person.role)}
                        </span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Face Recognition:</span>
                        <span className={`status-badge ${person.has_face_embeddings ? 'enabled' : 'disabled'}`}>
                          {person.has_face_embeddings ? '✓ Enabled' : '✗ Not Set'}
                        </span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Registered:</span>
                        <span className="detail-value">{formatDate(person.created_at)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default Employees

