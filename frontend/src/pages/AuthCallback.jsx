import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import './AuthCallback.css'

function AuthCallback() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showPasswordForm, setShowPasswordForm] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordError, setPasswordError] = useState('')

  useEffect(() => {
    const handleAuthCallback = async () => {
      try {
        // Get the hash from URL (contains access_token, refresh_token, etc.)
        const hash = window.location.hash
        const hashParams = new URLSearchParams(hash.substring(1))
        
        // Also check query params for type and tokens (sometimes Supabase puts them in query params)
        const queryType = searchParams.get('type')
        const hashType = hashParams.get('type')
        const type = queryType || hashType
        
        // Check both hash and query params for tokens
        const accessToken = hashParams.get('access_token') || searchParams.get('access_token')
        const refreshToken = hashParams.get('refresh_token') || searchParams.get('refresh_token')
        
        // Debug logging
        console.log('Auth callback - Type:', type)
        console.log('Auth callback - Has tokens:', !!accessToken && !!refreshToken)
        console.log('Auth callback - Hash:', hash)
        console.log('Auth callback - Full URL:', window.location.href)

        if (type === 'invite' || type === 'recovery') {
          // User needs to set password
          if (accessToken && refreshToken) {
            // Set the session first
            const { error: sessionError } = await supabase.auth.setSession({
              access_token: accessToken,
              refresh_token: refreshToken,
            })

            if (sessionError) {
              throw sessionError
            }

            // Check if user already has a password set
            const { data: { user } } = await supabase.auth.getUser()
            
            if (user) {
              // User is authenticated, now they need to set password
              setShowPasswordForm(true)
              setLoading(false)
              return
            }
          } else {
            // No tokens in URL, check if there's an existing session
            // This can happen if the user already clicked the link and has a session
            const { data: { session }, error: sessionError } = await supabase.auth.getSession()
            
            if (session && !sessionError) {
              // User has a session, allow them to set password
              setShowPasswordForm(true)
              setLoading(false)
              return
            } else {
              // No tokens and no session - for invite type, try to get session from storage
              // or show an error with option to request new invite
              if (type === 'invite') {
                // Try one more time to get session - sometimes it takes a moment
                await new Promise(resolve => setTimeout(resolve, 500))
                const { data: { session: retrySession } } = await supabase.auth.getSession()
                
                if (retrySession) {
                  setShowPasswordForm(true)
                  setLoading(false)
                  return
                }
                
                // Still no session - but for invite type, we'll show the form anyway
                // The user might be able to set password if Supabase allows it without explicit session
                // Or we'll show a helpful error when they try to submit
                console.warn('No session found for invite link - showing password form anyway')
                setShowPasswordForm(true)
                setLoading(false)
                return
              }
            }
          }
        } else if (accessToken && refreshToken) {
          // Regular auth callback - exchange tokens for session
          const { data, error: sessionError } = await supabase.auth.setSession({
            access_token: accessToken,
            refresh_token: refreshToken,
          })

          if (sessionError) throw sessionError

          if (data?.user) {
            // Successfully authenticated, redirect to dashboard
            navigate('/dashboard', { replace: true })
            return
          }
        } else {
          // No tokens found, might be an error
          const errorDescription = hashParams.get('error_description') || searchParams.get('error_description') || 'Authentication failed'
          setError(errorDescription)
          setLoading(false)
        }
      } catch (err) {
        console.error('Auth callback error:', err)
        setError(err.message || 'Failed to authenticate. Please try again.')
        setLoading(false)
      }
    }

    handleAuthCallback()
  }, [navigate, searchParams])

  const handleSetPassword = async (e) => {
    e.preventDefault()
    setPasswordError('')

    // Validate passwords
    if (!password || password.length < 6) {
      setPasswordError('Password must be at least 6 characters long')
      return
    }

    if (password !== confirmPassword) {
      setPasswordError('Passwords do not match')
      return
    }

    setIsSubmitting(true)

    try {
      // Check if user has a session first
      const { data: { session }, error: sessionError } = await supabase.auth.getSession()
      
      if (!session) {
        // No session - this is required for setting password
        // Try to get user info to see if there's any way to proceed
        const { data: { user }, error: userError } = await supabase.auth.getUser()
        
        if (!user) {
          throw new Error('No active session found. The invite link may have expired. Please request a new invite link or contact support.')
        }
        
        // If we have a user but no session, try to refresh
        console.warn('User found but no session - attempting to continue')
      }

      // Update password - this works for both authenticated users and invite flows
      const { error: updateError } = await supabase.auth.updateUser({
        password: password,
      })

      if (updateError) {
        // If update fails, it might be because user needs to be authenticated
        // Try to get the user info to see what's happening
        const { data: { user }, error: userError } = await supabase.auth.getUser()
        
        if (!user && updateError.message.includes('session')) {
          throw new Error('Session expired. Please click the invite link again.')
        }
        
        throw updateError
      }

      // Password set successfully, redirect to dashboard
      navigate('/dashboard', { replace: true })
    } catch (err) {
      console.error('Password update error:', err)
      setPasswordError(err.message || 'Failed to set password. Please try again.')
      setIsSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="auth-callback-container">
        <div className="auth-callback-card">
          <div className="spinner"></div>
          <p>Processing authentication...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="auth-callback-container">
        <div className="auth-callback-card error">
          <h2>Authentication Error</h2>
          <p>{error}</p>
          <button onClick={() => navigate('/login')} className="button-primary">
            Go to Login
          </button>
        </div>
      </div>
    )
  }

  if (showPasswordForm) {
    return (
      <div className="auth-callback-container">
        <div className="auth-callback-card">
          <h2>Set Your Password</h2>
          <p>Please set a password to complete your account setup.</p>
          
          <form onSubmit={handleSetPassword} className="password-form">
            {passwordError && (
              <div className="error-message">{passwordError}</div>
            )}

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
                minLength={6}
                disabled={isSubmitting}
              />
              <small>Must be at least 6 characters</small>
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                id="confirmPassword"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm your password"
                required
                minLength={6}
                disabled={isSubmitting}
              />
            </div>

            <button type="submit" className="button-primary" disabled={isSubmitting}>
              {isSubmitting ? 'Setting Password...' : 'Set Password'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  return null
}

export default AuthCallback

