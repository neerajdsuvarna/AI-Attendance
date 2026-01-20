import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import { io } from 'socket.io-client'
import './LiveDetection.css'

// Backend API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function LiveDetection({ userRole }) {
  const navigate = useNavigate()
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const frameIntervalRef = useRef(null)
  const socketRef = useRef(null)
  const sendingFramesRef = useRef(false)
  
  const [isActive, setIsActive] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const [detectedPerson, setDetectedPerson] = useState(null)
  const [allDetections, setAllDetections] = useState([]) // Store all face detections
  const [recentActivity, setRecentActivity] = useState([])
  const [isConnected, setIsConnected] = useState(false)
  const cameraRetryCountRef = useRef(0)
  const MAX_RETRIES = 3
  const drawIntervalRef = useRef(null)
  const animationFrameRef = useRef(null)

  // Frame capture function - optimized for faster transmission
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      return null
    }

    const canvas = document.createElement('canvas')
    // Resize to max 960px width for faster network transfer (Raspberry Pi compatible size)
    // Face detection models work fine at this resolution
    const maxWidth = 960
    const videoWidth = videoRef.current.videoWidth
    const videoHeight = videoRef.current.videoHeight
    const scale = Math.min(1, maxWidth / videoWidth)
    
    canvas.width = videoWidth * scale
    canvas.height = videoHeight * scale
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    // Reduce quality to 0.5 for faster transmission (still good quality for face detection)
    return canvas.toDataURL('image/jpeg', 0.5)
  }, [])

  // Start camera when component mounts
  useEffect(() => {
    const startCamera = async (retryCount = 0) => {
      try {
        // Wait for video element to be mounted
        if (!videoRef.current) {
          console.log('⏳ Waiting for video element to mount...')
          setTimeout(() => {
            if (retryCount < MAX_RETRIES) {
              startCamera(retryCount + 1)
            } else {
              setError('Video element not found. Please refresh the page.')
              setIsLoading(false)
            }
          }, 500)
          return
        }

        console.log('🎥 Requesting camera access...')
        setIsLoading(true)
        setError(null)

        // Stop any existing stream first
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop())
        }

        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          }, 
          audio: false
        })

        // Double-check video element is still available
        if (!videoRef.current) {
          stream.getTracks().forEach(track => track.stop())
          throw new Error('Video element was removed')
        }

        // Set stream and wait for video to be ready
        videoRef.current.srcObject = stream
        streamRef.current = stream

        // Wait for video to actually load
        await new Promise((resolve, reject) => {
          if (!videoRef.current) {
            reject(new Error('Video element not available'))
            return
          }

          const video = videoRef.current
          
          const handleLoadedMetadata = () => {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata)
            video.removeEventListener('error', handleError)
            console.log('✅ Camera stream loaded successfully')
            setIsLoading(false)
            setError(null)
            cameraRetryCountRef.current = 0
            resolve()
          }

          const handleError = (e) => {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata)
            video.removeEventListener('error', handleError)
            reject(new Error('Video element failed to load stream'))
          }

          if (video.readyState >= 1) {
            handleLoadedMetadata()
          } else {
            video.addEventListener('loadedmetadata', handleLoadedMetadata)
            video.addEventListener('error', handleError)
            
            setTimeout(() => {
              video.removeEventListener('loadedmetadata', handleLoadedMetadata)
              video.removeEventListener('error', handleError)
              reject(new Error('Video load timeout'))
            }, 5000)
          }
        })

      } catch (err) {
        console.error('❌ Error accessing camera:', err)
        cameraRetryCountRef.current = retryCount + 1

        let errorMessage = 'Failed to access camera. '
        
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          errorMessage += 'Please allow camera permissions and refresh the page.'
          setError(errorMessage)
          setIsLoading(false)
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          errorMessage += 'No camera found. Please connect a camera and refresh the page.'
          setError(errorMessage)
          setIsLoading(false)
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          errorMessage += 'Camera is being used by another application. Please close other apps and refresh.'
          setError(errorMessage)
          setIsLoading(false)
        } else if (retryCount < MAX_RETRIES) {
          console.log(`🔄 Retrying camera access (attempt ${retryCount + 1}/${MAX_RETRIES})...`)
          setTimeout(() => {
            startCamera(retryCount + 1)
          }, 1000 * (retryCount + 1))
        } else {
          errorMessage += 'Please refresh the page and try again.'
          setError(errorMessage)
          setIsLoading(false)
        }
      }
    }

    startCamera()

    return () => {
      // Stop everything in proper order
      sendingFramesRef.current = false
      stopFrameSending()
      stopDetection()
      stopCamera()
      
      // Clean up socket
      if (socketRef.current) {
        socketRef.current.removeAllListeners()
        if (socketRef.current.connected) {
          socketRef.current.disconnect()
        }
        socketRef.current = null
      }
    }
  }, [])

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  // Initialize Socket.IO connection
  useEffect(() => {
    if (!isActive) {
      if (socketRef.current) {
        socketRef.current.disconnect()
        socketRef.current = null
      }
      setIsConnected(false)
      return
    }

    const initSocket = async () => {
      // Get auth token
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        setError('Not authenticated. Please log in again.')
        setIsActive(false)
        return
      }

      console.log('🔗 Initializing socket connection for face detection')
      const socket = io(API_BASE_URL, {
        transports: ['websocket', 'polling'],
        timeout: 20000,
        forceNew: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        auth: {
          token: session.access_token
        }
      })

      socket.on('connect', () => {
        console.log('🔗 Socket connected for face detection')
        setIsConnected(true)
        setError(null)
        // Small delay to ensure socket is fully ready before sending frames
        setTimeout(() => {
          if (socketRef.current && socketRef.current.connected && isActive) {
            startFrameSending(session.access_token)
          }
        }, 100)
      })

      socket.on('disconnect', (reason) => {
        console.log('🔌 Socket disconnected:', reason)
        setIsConnected(false)
        sendingFramesRef.current = false
        stopFrameSending()
      })

      socket.on('connect_error', (error) => {
        console.error('🔌 Socket connection error:', error)
        setError('Connection failed - Backend may not be running')
        setIsConnected(false)
        sendingFramesRef.current = false
        stopFrameSending()
      })

      socket.on('detection_response', (data) => {
        if (data.error) {
          console.error('Detection error:', data.error)
          setAllDetections([])
          return
        }

        if (data.success && data.detections) {
          // Performance logging (remove in production)
          const responseTime = performance.now()
          if (window.lastFrameTime) {
            const latency = responseTime - window.lastFrameTime
            if (latency > 200) {
              console.log(`[PERF] Frame processing took ${latency.toFixed(0)}ms`)
            }
          }
          window.lastFrameTime = responseTime
          
          // Store all detections for drawing bounding boxes (immediate update for faster rendering)
          setAllDetections(data.detections || [])
          
          // Find the best recognized match for the badge
          if (data.employees && data.employees.length > 0) {
            const bestMatch = data.employees[0]
            setDetectedPerson({
              id: bestMatch.employee_id,
              name: bestMatch.name,
              similarity: bestMatch.similarity
            })
            
            // Add to recent activity (only for recognized faces)
            // Generate unique ID using timestamp + random number to avoid duplicate keys
            const activity = {
              id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              name: bestMatch.name,
              similarity: bestMatch.similarity,
              timestamp: new Date().toLocaleTimeString()
            }
            setRecentActivity(prev => {
              // Avoid duplicates in quick succession (same person within 2 seconds)
              const exists = prev.some(a => a.name === activity.name && 
                Math.abs(new Date(a.timestamp) - new Date(activity.timestamp)) < 2000)
              if (exists) return prev
              return [activity, ...prev.slice(0, 9)]
            })
            setError(null)
          } else {
            setDetectedPerson(null)
          }
        } else {
          setAllDetections([])
          setDetectedPerson(null)
        }
      })

      socketRef.current = socket
    }

    initSocket()

    return () => {
      // Stop frame sending first
      sendingFramesRef.current = false
      stopFrameSending()
      
      // Then disconnect socket
      if (socketRef.current) {
        console.log('🔌 Cleaning up socket connection')
        // Remove all listeners to prevent memory leaks
        socketRef.current.removeAllListeners()
        // Disconnect if still connected
        if (socketRef.current.connected) {
          socketRef.current.disconnect()
        }
        socketRef.current = null
      }
    }
  }, [isActive])

  const startFrameSending = useCallback(async (authToken) => {
    if (!isActive || !socketRef.current) return
    
    // Verify socket is actually connected before starting
    if (!socketRef.current.connected) {
      console.log('📡 Socket not connected yet, waiting...')
      return
    }
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current)
      frameIntervalRef.current = null
    }

    console.log('📡 Starting frame sending...')
    sendingFramesRef.current = true
    
    frameIntervalRef.current = setInterval(async () => {
      // Check if we should still be sending
      if (!sendingFramesRef.current || !socketRef.current) {
        return
      }

      // Check if socket is actually connected
      if (!socketRef.current.connected) {
        console.log('📡 Socket not connected, stopping frame sending')
        stopFrameSending()
        return
      }

      const img = captureFrame()
      if (!img) {
        return
      }

      // Get fresh auth token
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || authToken

      // Check socket state before attempting to emit
      if (!socketRef.current || !socketRef.current.connected) {
        stopFrameSending()
        return
      }

      // Performance logging
      window.lastFrameTime = performance.now()
      
      // Emit without try-catch - let Socket.IO handle errors internally
      // The socket library will handle connection state automatically
      socketRef.current.emit('detection_frame', { 
        image: img,
        auth_token: token
      })
    }, 100) // Send frame every 100ms (10 FPS) - increased from 200ms for faster response
  }, [isActive, captureFrame])

  const stopFrameSending = useCallback(() => {
    sendingFramesRef.current = false
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current)
      frameIntervalRef.current = null
      console.log('📡 Frame sending stopped')
    }
  }, [])

  const stopDetection = () => {
    stopFrameSending()
    setDetectedPerson(null)
    setAllDetections([])
    if (drawIntervalRef.current) {
      clearInterval(drawIntervalRef.current)
      drawIntervalRef.current = null
    }
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }

  // Draw bounding boxes on canvas overlay
  useEffect(() => {
    if (!isActive || !videoRef.current || !canvasRef.current || allDetections.length === 0) {
      // Clear canvas if not active or no detections
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d')
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      }
      return
    }

    let lastVideoWidth = 0
    let lastVideoHeight = 0
    let lastDisplayWidth = 0
    let lastDisplayHeight = 0

    const drawBoxes = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      
      if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
        return
      }

      // Get video display dimensions
      const videoRect = video.getBoundingClientRect()
      const videoDisplayWidth = videoRect.width
      const videoDisplayHeight = videoRect.height
      
      // Only resize canvas when dimensions actually change (MAJOR PERFORMANCE FIX!)
      if (canvas.width !== videoDisplayWidth || canvas.height !== videoDisplayHeight ||
          lastVideoWidth !== video.videoWidth || lastVideoHeight !== video.videoHeight) {
        canvas.width = videoDisplayWidth
        canvas.height = videoDisplayHeight
        lastVideoWidth = video.videoWidth
        lastVideoHeight = video.videoHeight
        lastDisplayWidth = videoDisplayWidth
        lastDisplayHeight = videoDisplayHeight
      }
      
      // Calculate scale factors
      const scaleX = videoDisplayWidth / video.videoWidth
      const scaleY = videoDisplayHeight / video.videoHeight

      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw all detected faces
      allDetections.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox || []
        if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
          return // Skip invalid bboxes
        }

        // Scale bounding box to match video display size
        const scaledX1 = x1 * scaleX
        const scaledY1 = y1 * scaleY
        const scaledX2 = x2 * scaleX
        const scaledY2 = y2 * scaleY
        const width = scaledX2 - scaledX1
        const height = scaledY2 - scaledY1

        // Mirror X coordinates horizontally to match mirrored video display
        // Video is displayed mirrored (like a mirror), but backend coordinates are not mirrored
        const mirroredX1 = canvas.width - scaledX2
        const mirroredX2 = canvas.width - scaledX1

        // Choose color based on recognition status
        if (detection.recognized) {
          // Green for recognized employees
          ctx.strokeStyle = '#10b981'
          ctx.fillStyle = 'rgba(16, 185, 129, 0.2)'
          ctx.lineWidth = 3
        } else {
          // Yellow for unrecognized faces
          ctx.strokeStyle = '#f59e0b'
          ctx.fillStyle = 'rgba(245, 158, 11, 0.2)'
          ctx.lineWidth = 2
        }

        // Draw filled rectangle (using mirrored X coordinates)
        ctx.fillRect(mirroredX1, scaledY1, width, height)
        
        // Draw border
        ctx.strokeRect(mirroredX1, scaledY1, width, height)

        // Draw label
        if (detection.recognized && detection.employee_name) {
          ctx.font = 'bold 16px Arial'
          ctx.textBaseline = 'top'
          
          const label = `${detection.employee_name} (${(detection.similarity * 100).toFixed(1)}%)`
          const textWidth = ctx.measureText(label).width
          
          // Draw label background (using mirrored X coordinate)
          ctx.fillStyle = 'rgba(16, 185, 129, 0.9)'
          ctx.fillRect(mirroredX1, scaledY1 - 25, textWidth + 10, 22)
          
          // Draw label text
          ctx.fillStyle = 'white'
          ctx.fillText(label, mirroredX1 + 5, scaledY1 - 22)
        } else {
          // Draw "Unknown" label for unrecognized faces
          ctx.font = 'bold 14px Arial'
          ctx.textBaseline = 'top'
          
          const label = 'Unknown'
          const textWidth = ctx.measureText(label).width
          
          // Draw label background (using mirrored X coordinate)
          ctx.fillStyle = 'rgba(245, 158, 11, 0.9)'
          ctx.fillRect(mirroredX1, scaledY1 - 22, textWidth + 10, 20)
          
          // Draw label text
          ctx.fillStyle = 'white'
          ctx.fillText(label, mirroredX1 + 5, scaledY1 - 20)
        }
      })
    }

    // Draw boxes immediately and use requestAnimationFrame for smooth rendering
    drawBoxes() // Draw immediately
    
    const animate = () => {
      drawBoxes()
      animationFrameRef.current = requestAnimationFrame(animate)
    }
    animationFrameRef.current = requestAnimationFrame(animate)
    
    // Fallback interval in case requestAnimationFrame doesn't work
    drawIntervalRef.current = setInterval(drawBoxes, 100) // Update every 100ms as backup

    return () => {
      if (drawIntervalRef.current) {
        clearInterval(drawIntervalRef.current)
        drawIntervalRef.current = null
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }
  }, [isActive, allDetections])

  const handleToggleDetection = () => {
    if (isActive) {
      setIsActive(false)
      stopDetection()
    } else {
      setIsActive(true)
      // Socket will be initialized in useEffect when isActive becomes true
    }
  }

  return (
    <div className="live-detection-container">
      <div className="live-detection-header">
        <button 
          onClick={() => {
            stopDetection()
            stopCamera()
            navigate('/dashboard')
          }}
          className="back-button"
        >
          ← Back to Dashboard
        </button>
        <h1>Live Face Detection</h1>
        <p>Real-time attendance tracking using face recognition</p>
      </div>

      <div className="live-detection-content">
        <div className="detection-status-card">
          <div className={`status-indicator ${isActive ? 'active' : 'inactive'}`}>
            <div className="status-pulse"></div>
            <span className="status-text">
              {isActive ? '🔴 Live' : '⚪ Inactive'}
            </span>
          </div>
          <h2>{isActive ? 'Detection Active' : 'Detection Inactive'}</h2>
          <p>
            {isActive 
              ? 'Face recognition is currently running. Employees can check in/out by facing the camera.'
              : 'Click "Start Detection" to begin real-time face recognition for attendance tracking.'
            }
          </p>
        </div>

        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}

        <div className="detection-controls">
          <button
            onClick={handleToggleDetection}
            className={`control-button ${isActive ? 'stop-button' : 'start-button'}`}
            disabled={isLoading}
          >
            {isLoading ? '⏳ Loading...' : isActive ? '⏹ Stop Detection' : '▶ Start Detection'}
          </button>
        </div>

        <div className="detection-preview">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="camera-video"
          />
          <canvas 
            ref={canvasRef} 
            className="detection-canvas"
            style={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 5
            }} 
          />
          
          {isLoading && (
            <div className="preview-overlay">
              <div className="loading-spinner"></div>
              <p>Starting camera...</p>
            </div>
          )}
          
          {!isLoading && !isActive && videoRef.current?.srcObject && (
            <div className="preview-overlay">
              <p>Camera ready. Click "Start Detection" to begin.</p>
            </div>
          )}
          
          {detectedPerson && (
            <div className="detected-person-overlay">
              <div className="detected-badge">
                <span className="detected-icon">✅</span>
                <div className="detected-info">
                  <p className="detected-name">{detectedPerson.name}</p>
                  <p className="detected-similarity">
                    {(detectedPerson.similarity * 100).toFixed(1)}% match
                  </p>
                </div>
              </div>
            </div>
          )}
          
          {!videoRef.current?.srcObject && !isLoading && (
            <div className="preview-placeholder">
              <div className="camera-icon-large">📷</div>
              <p>Camera Preview</p>
              <p className="preview-hint">
                Waiting for camera...
              </p>
            </div>
          )}
        </div>

        <div className="detection-info">
          <h3>How it works:</h3>
          <ul>
            <li>📸 Start detection to activate the camera</li>
            <li>👤 Employees face the camera to check in or check out</li>
            <li>✅ Face recognition automatically identifies the employee</li>
            <li>📝 Attendance is recorded in real-time</li>
            <li>🔔 You'll see notifications for successful check-ins/outs</li>
          </ul>
        </div>

        <div className="recent-activity">
          <h3>Recent Activity</h3>
          {recentActivity.length === 0 ? (
            <div className="activity-placeholder">
              <p>No recent activity. Start detection to see check-ins and check-outs.</p>
            </div>
          ) : (
            <div className="activity-list">
              {recentActivity.map((activity) => (
                <div key={activity.id} className="activity-item">
                  <div className="activity-icon">✅</div>
                  <div className="activity-details">
                    <p className="activity-name">{activity.name}</p>
                    <p className="activity-meta">
                      Detected at {activity.timestamp} • {(activity.similarity * 100).toFixed(1)}% match
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default LiveDetection

