import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useState, useEffect, useCallback } from 'react'
import { supabase } from './lib/supabase'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import RegisterEmployee from './pages/RegisterEmployee'
import LiveDetection from './pages/LiveDetection'
import Employees from './pages/Employees'
import Attendance from './pages/Attendance'
import AuthCallback from './pages/AuthCallback'
import './App.css'

function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [userRole, setUserRole] = useState(null)

  const fetchUserRole = useCallback(async (userId) => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', userId)
        .single()

      if (error) throw error
      setUserRole(data?.role || null)
    } catch (error) {
      console.error('Error fetching user role:', error)
      setUserRole(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let mounted = true

    // Check active session with error handling
    supabase.auth.getSession()
      .then(({ data: { session }, error }) => {
        if (!mounted) return
        
        if (error) {
          console.warn('Session error (this is normal if not logged in):', error.message)
          // Don't sign out automatically - just clear the state
          setUser(null)
          setUserRole(null)
          setLoading(false)
          return
        }
        
        if (session?.user) {
          setUser(session.user)
          fetchUserRole(session.user.id)
        } else {
          setUser(null)
          setUserRole(null)
          setLoading(false)
        }
      })
      .catch((error) => {
        if (!mounted) return
        console.warn('Session check failed:', error.message)
        setUser(null)
        setUserRole(null)
        setLoading(false)
      })

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!mounted) return
      
      if (session?.user) {
        setUser(session.user)
        fetchUserRole(session.user.id)
      } else {
        setUser(null)
        setUserRole(null)
        setLoading(false)
      }
    })

    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [fetchUserRole])

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    )
  }

  // Prevent navigation loops - only navigate when not loading
  return (
    <Router>
      <Routes>
        <Route
          path="/login"
          element={user && userRole ? <Navigate to="/dashboard" replace /> : <Login />}
        />
        <Route
          path="/dashboard"
          element={
            user && userRole ? (
              <Dashboard user={user} userRole={userRole} />
            ) : user ? (
              <div className="loading-container">
                <div className="spinner"></div>
                <p>Loading role...</p>
              </div>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
        <Route
          path="/register-employee"
          element={
            user && (userRole === 'super_admin' || userRole === 'hr_admin') ? (
              <RegisterEmployee userRole={userRole} />
            ) : (
              <Navigate to="/dashboard" replace />
            )
          }
        />
        <Route
          path="/live-detection"
          element={
            user && (userRole === 'super_admin' || userRole === 'hr_admin') ? (
              <LiveDetection userRole={userRole} />
            ) : (
              <Navigate to="/dashboard" replace />
            )
          }
        />
        <Route
          path="/employees"
          element={
            user && userRole ? (
              <Employees userRole={userRole} />
            ) : (
              <Navigate to="/dashboard" replace />
            )
          }
        />
        <Route
          path="/attendance"
          element={
            user && userRole ? (
              <Attendance userRole={userRole} />
            ) : (
              <Navigate to="/dashboard" replace />
            )
          }
        />
        <Route
          path="/auth/callback"
          element={<AuthCallback />}
        />
        <Route 
          path="/" 
          element={
            <Navigate 
              to={user && userRole ? "/dashboard" : "/login"} 
              replace 
            />
          } 
        />
      </Routes>
    </Router>
  )
}

export default App