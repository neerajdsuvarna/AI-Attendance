import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import './Attendance.css'

function Attendance({ userRole }) {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [attendance, setAttendance] = useState([])
  const [summary, setSummary] = useState({
    total_records: 0,
    present_today: 0,
    exited_today: 0,
    total_today: 0
  })
  
  // Filters
  const [dateFilter, setDateFilter] = useState('today') // 'today', 'week', 'month', 'custom', 'all'
  const [statusFilter, setStatusFilter] = useState('all') // 'all', 'present', 'exited'
  const [customStartDate, setCustomStartDate] = useState('')
  const [customEndDate, setCustomEndDate] = useState('')
  const [selectedEmployee, setSelectedEmployee] = useState('all')
  const [employees, setEmployees] = useState([])

  const isAdmin = userRole === 'super_admin' || userRole === 'hr_admin'

  useEffect(() => {
    fetchEmployees()
    fetchAttendance()
  }, [])

  useEffect(() => {
    fetchAttendance()
  }, [dateFilter, statusFilter, customStartDate, customEndDate, selectedEmployee])

  const fetchEmployees = async () => {
    if (!isAdmin) return

    try {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return

      const { data, error: fetchError } = await supabase.functions.invoke('get-employees-list', {
        headers: {
          Authorization: `Bearer ${session.access_token}`,
        },
      })

      if (!fetchError && data?.success) {
        const allEmployees = [...(data.hr_admins || []), ...(data.employees || [])]
        setEmployees(allEmployees)
      }
    } catch (err) {
      console.error('Error fetching employees:', err)
    }
  }

  const fetchAttendance = async () => {
    try {
      setLoading(true)
      setError(null)

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        setError('Not authenticated. Please log in again.')
        return
      }

      // Build query parameters
      const params = new URLSearchParams()
      
      if (dateFilter === 'today') {
        const today = new Date().toISOString().split('T')[0]
        params.append('date', today)
      } else if (dateFilter === 'week') {
        const today = new Date()
        const weekStart = new Date(today)
        weekStart.setDate(today.getDate() - today.getDay()) // Start of week (Sunday)
        const weekEnd = new Date(weekStart)
        weekEnd.setDate(weekStart.getDate() + 6) // End of week
        
        params.append('start_date', weekStart.toISOString().split('T')[0])
        params.append('end_date', weekEnd.toISOString().split('T')[0])
      } else if (dateFilter === 'month') {
        const today = new Date()
        const monthStart = new Date(today.getFullYear(), today.getMonth(), 1)
        const monthEnd = new Date(today.getFullYear(), today.getMonth() + 1, 0)
        
        params.append('start_date', monthStart.toISOString().split('T')[0])
        params.append('end_date', monthEnd.toISOString().split('T')[0])
      } else if (dateFilter === 'custom' && customStartDate && customEndDate) {
        params.append('start_date', customStartDate)
        params.append('end_date', customEndDate)
      }
      // 'all' doesn't add date params

      if (statusFilter !== 'all') {
        params.append('status', statusFilter)
      }

      if (isAdmin && selectedEmployee !== 'all') {
        params.append('employee_id', selectedEmployee)
      }

      // Build URL with query parameters
      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'http://127.0.0.1:54321'
      const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0'
      
      const baseUrl = `${supabaseUrl}/functions/v1/get-attendance`
      const queryString = params.toString()
      const url = queryString ? `${baseUrl}?${queryString}` : baseUrl

      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'apikey': supabaseKey,
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()

      if (result.error) {
        throw new Error(result.error)
      }

      if (result.success) {
        setAttendance(result.attendance || [])
        setSummary(result.summary || {
          total_records: 0,
          present_today: 0,
          exited_today: 0,
          total_today: 0
        })
      } else {
        throw new Error('Failed to fetch attendance')
      }
    } catch (err) {
      console.error('Error fetching attendance:', err)
      setError(err.message || 'Failed to load attendance. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const formatTime = (dateString) => {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    })
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

  const getStatusBadge = (status) => {
    if (status === 'present') {
      return (
        <span className="status-badge present">
          <span className="status-dot"></span>
          Present
        </span>
      )
    } else {
      return (
        <span className="status-badge exited">
          <span className="status-dot"></span>
          Exited
        </span>
      )
    }
  }

  return (
    <div className="attendance-container">
      <div className="attendance-header">
        <button
          onClick={() => navigate('/dashboard')}
          className="back-button"
        >
          ← Back to Dashboard
        </button>
        <h1>Attendance Records</h1>
        <p>View and track employee attendance, check-ins, and check-outs</p>
      </div>

      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={fetchAttendance} className="retry-button">
            Retry
          </button>
        </div>
      )}

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="summary-icon total">📊</div>
          <div className="summary-content">
            <h3>{summary.total_records}</h3>
            <p>Total Records</p>
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-icon present">✅</div>
          <div className="summary-content">
            <h3>{summary.present_today}</h3>
            <p>Present Today</p>
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-icon exited">🚪</div>
          <div className="summary-content">
            <h3>{summary.exited_today}</h3>
            <p>Exited Today</p>
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-icon today">📅</div>
          <div className="summary-content">
            <h3>{summary.total_today}</h3>
            <p>Total Today</p>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="filters-section">
        <div className="filters-row">
          <div className="filter-group">
            <label>Date Range</label>
            <select 
              value={dateFilter} 
              onChange={(e) => setDateFilter(e.target.value)}
              className="filter-select"
            >
              <option value="today">Today</option>
              <option value="week">This Week</option>
              <option value="month">This Month</option>
              <option value="custom">Custom Range</option>
              <option value="all">All Time</option>
            </select>
          </div>

          {dateFilter === 'custom' && (
            <div className="filter-group">
              <label>Start Date</label>
              <input
                type="date"
                value={customStartDate}
                onChange={(e) => setCustomStartDate(e.target.value)}
                className="filter-input"
              />
            </div>
          )}

          {dateFilter === 'custom' && (
            <div className="filter-group">
              <label>End Date</label>
              <input
                type="date"
                value={customEndDate}
                onChange={(e) => setCustomEndDate(e.target.value)}
                className="filter-input"
              />
            </div>
          )}

          <div className="filter-group">
            <label>Status</label>
            <select 
              value={statusFilter} 
              onChange={(e) => setStatusFilter(e.target.value)}
              className="filter-select"
            >
              <option value="all">All</option>
              <option value="present">Present</option>
              <option value="exited">Exited</option>
            </select>
          </div>

          {isAdmin && (
            <div className="filter-group">
              <label>Employee</label>
              <select 
                value={selectedEmployee} 
                onChange={(e) => setSelectedEmployee(e.target.value)}
                className="filter-select"
              >
                <option value="all">All Employees</option>
                {employees.map(emp => (
                  <option key={emp.id} value={emp.id}>{emp.name}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Attendance Table */}
      {loading ? (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading attendance records...</p>
        </div>
      ) : (
        <div className="attendance-table-container">
          {attendance.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📋</div>
              <h3>No attendance records found</h3>
              <p>
                {dateFilter === 'today'
                  ? 'No attendance records for today.'
                  : 'No attendance records match your filters.'}
              </p>
            </div>
          ) : (
            <table className="attendance-table">
              <thead>
                <tr>
                  <th>Employee</th>
                  <th>Entry Time</th>
                  <th>Exit Time</th>
                  <th>Hours Worked</th>
                  <th>Status</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {attendance.map((record) => (
                  <tr key={record.id}>
                    <td>
                      <div className="employee-cell">
                        <div className="employee-avatar-small">
                          {record.employee_name.charAt(0).toUpperCase()}
                        </div>
                        <div>
                          <div className="employee-name-cell">{record.employee_name}</div>
                          {isAdmin && (
                            <div className="employee-email-cell">{record.employee_email}</div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="time-cell">
                        <span className="time-value">{formatTime(record.entry_time)}</span>
                        <span className="time-date">{formatDate(record.entry_time)}</span>
                      </div>
                    </td>
                    <td>
                      {record.exit_time ? (
                        <div className="time-cell">
                          <span className="time-value">{formatTime(record.exit_time)}</span>
                          <span className="time-date">{formatDate(record.exit_time)}</span>
                        </div>
                      ) : (
                        <span className="no-exit">—</span>
                      )}
                    </td>
                    <td>
                      {record.hours_worked ? (
                        <span className="hours-worked">{record.hours_worked} hrs</span>
                      ) : (
                        <span className="hours-calculating">Calculating...</span>
                      )}
                    </td>
                    <td>{getStatusBadge(record.status)}</td>
                    <td>{formatDate(record.entry_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  )
}

export default Attendance

