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
  const sentImageDimensionsRef = useRef({ width: 0, height: 0 }) // Store dimensions of image sent to backend
  const detectionConfigRef = useRef({ maxWidth: 960, quality: 0.5 }) // Detection config from backend
  const cameraMaxResolutionRef = useRef({ width: 0, height: 0 }) // Store camera's maximum resolution
  
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

  // Detect camera maximum capabilities
  const detectCameraCapabilities = useCallback(async () => {
    try {
      // Try to get capabilities using getCapabilities() (modern browsers)
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' },
        audio: false 
      })
      
      const videoTrack = stream.getVideoTracks()[0]
      if (videoTrack && videoTrack.getCapabilities) {
        const capabilities = videoTrack.getCapabilities()
        if (capabilities.width && capabilities.height) {
          const maxWidth = capabilities.width.max || 1920
          const maxHeight = capabilities.height.max || 1080
          cameraMaxResolutionRef.current = { width: maxWidth, height: maxHeight }
          console.log(`[INFO] Camera max resolution detected: ${maxWidth}x${maxHeight}`)
          
          // Send camera capabilities to backend
          await sendCameraCapabilitiesToBackend(maxWidth, maxHeight)
          
          // Stop the test stream
          stream.getTracks().forEach(track => track.stop())
          return { width: maxWidth, height: maxHeight }
        }
      }
      
      // Fallback: Try to detect by checking actual video dimensions
      // Request high resolution and see what we get
      const testStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 3840, max: 3840 },
          height: { ideal: 2160, max: 2160 },
          facingMode: 'user'
        },
        audio: false
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = testStream
        await new Promise((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => {
              const actualWidth = videoRef.current.videoWidth
              const actualHeight = videoRef.current.videoHeight
              cameraMaxResolutionRef.current = { width: actualWidth, height: actualHeight }
              console.log(`[INFO] Camera max resolution detected (fallback): ${actualWidth}x${actualHeight}`)
              
              // Send camera capabilities to backend
              sendCameraCapabilitiesToBackend(actualWidth, actualHeight)
              
              testStream.getTracks().forEach(track => track.stop())
              resolve()
            }
          } else {
            resolve()
          }
        })
      } else {
        testStream.getTracks().forEach(track => track.stop())
      }
      
      return cameraMaxResolutionRef.current
    } catch (error) {
      console.warn('[WARNING] Failed to detect camera capabilities, using defaults:', error)
      // Default to reasonable values
      cameraMaxResolutionRef.current = { width: 1920, height: 1080 }
      return cameraMaxResolutionRef.current
    }
  }, [])

  // Send camera capabilities to backend
  const sendCameraCapabilitiesToBackend = useCallback(async (maxWidth, maxHeight) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/face/set-camera-capabilities`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          max_width: maxWidth,
          max_height: maxHeight
        })
      })
      
      if (response.ok) {
        const result = await response.json()
        if (result.success) {
          console.log(`[INFO] Camera capabilities sent to backend: ${maxWidth}x${maxHeight}`)
          // Update detection config after backend processes camera capabilities
          await fetchDetectionConfig()
        }
      }
    } catch (error) {
      console.warn('[WARNING] Failed to send camera capabilities to backend:', error)
    }
  }, [])

  // Fetch detection configuration from backend
  const fetchDetectionConfig = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/face/detection-config`)
      if (response.ok) {
        const config = await response.json()
        if (config.success) {
          detectionConfigRef.current = {
            maxWidth: config.recommended_capture.max_width,
            quality: config.recommended_capture.quality
          }
          console.log(`[INFO] Detection config loaded: maxWidth=${config.recommended_capture.max_width}, quality=${config.recommended_capture.quality}, GPU=${config.gpu_available}`)
        }
      }
    } catch (error) {
      console.warn('[WARNING] Failed to fetch detection config, using defaults:', error)
    }
  }, [])

  // Frame capture function - optimized based on backend GPU/CPU capabilities
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      return null
    }

    const canvas = document.createElement('canvas')
    const config = detectionConfigRef.current
    const maxWidth = config.maxWidth || 960 // Fallback to 960 if config not loaded
    const quality = config.quality || 0.5  // Fallback to 0.5 if config not loaded
    
    const videoWidth = videoRef.current.videoWidth
    const videoHeight = videoRef.current.videoHeight
    const scale = Math.min(1, maxWidth / videoWidth)
    
    const sentWidth = videoWidth * scale
    const sentHeight = videoHeight * scale
    
    // Store sent image dimensions for coordinate scaling
    // Backend returns coordinates in this sent image's coordinate space
    sentImageDimensionsRef.current = { width: sentWidth, height: sentHeight }
    
    canvas.width = sentWidth
    canvas.height = sentHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    // Use quality from backend config (higher for GPU, lower for CPU)
    return canvas.toDataURL('image/jpeg', quality)
  }, [])

  // Fetch detection config on mount
  useEffect(() => {
    fetchDetectionConfig()
  }, [fetchDetectionConfig])

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

        // Detect camera capabilities first
        const cameraMax = await detectCameraCapabilities()
        const cameraMaxWidth = cameraMax?.width || 1920
        const cameraMaxHeight = cameraMax?.height || 1080
        
        // Fetch detection config to determine optimal camera resolution
        await fetchDetectionConfig()
        const config = detectionConfigRef.current
        const backendMaxWidth = config.maxWidth || 1920
        
        // Use the minimum of camera max and backend recommended max
        // This ensures we don't exceed camera capabilities or backend processing limits
        const idealWidth = Math.min(cameraMaxWidth, backendMaxWidth)
        const idealHeight = Math.min(cameraMaxHeight, Math.round(idealWidth * 9 / 16))
        
        console.log(`[INFO] Requesting camera at: ${idealWidth}x${idealHeight} (camera max: ${cameraMaxWidth}x${cameraMaxHeight}, backend max: ${backendMaxWidth})`)

        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: idealWidth, max: cameraMaxWidth },
            height: { ideal: idealHeight, max: cameraMaxHeight },
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
        // Fetch latest detection config when socket connects
        fetchDetectionConfig().then(() => {
          // Small delay to ensure socket is fully ready before sending frames
          setTimeout(() => {
            if (socketRef.current && socketRef.current.connected && isActive) {
              startFrameSending(session.access_token)
            }
          }, 100)
        })
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
    }, 300) // Send frame every 300ms (~3.3 FPS) - optimized to match backend processing speed
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
      // Backend returns coordinates in the SENT image's coordinate space (resized to 960px)
      // We need to scale from sent image dimensions to display dimensions, not from original video dimensions
      const sentWidth = sentImageDimensionsRef.current.width || video.videoWidth
      const sentHeight = sentImageDimensionsRef.current.height || video.videoHeight
      
      // Scale from sent image space to display space
      const scaleX = videoDisplayWidth / sentWidth
      const scaleY = videoDisplayHeight / sentHeight

      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw all detected faces
      allDetections.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox || []
        if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) {
          return // Skip invalid bboxes
        }

        // Scale bounding box from sent image coordinates to video display size
        // Backend coordinates are in the sent image's space (e.g., 960px width)
        const scaledX1 = x1 * scaleX
        const scaledY1 = y1 * scaleY
        const scaledX2 = x2 * scaleX
        const scaledY2 = y2 * scaleY
        const width = scaledX2 - scaledX1
        const height = scaledY2 - scaledY1

        // Mirror X coordinates horizontally to match mirrored video display
        // Video is displayed mirrored via CSS transform: scaleX(-1)
        // Backend coordinates are in original (non-mirrored) space
        // To mirror: newX = canvasWidth - oldX
        // For a box: mirror the right edge to become left, left edge to become right
        const mirroredX1 = canvas.width - scaledX2  // Right edge becomes left edge
        const mirroredX2 = canvas.width - scaledX1  // Left edge becomes right edge
        const mirroredWidth = mirroredX2 - mirroredX1  // Should equal width

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

        // Draw filled rectangle using mirrored coordinates
        ctx.fillRect(mirroredX1, scaledY1, mirroredWidth, height)
        
        // Draw border
        ctx.strokeRect(mirroredX1, scaledY1, mirroredWidth, height)

        // Draw label
        if (detection.recognized && detection.employee_name) {
          ctx.font = 'bold 16px Arial'
          ctx.textBaseline = 'top'
          
          const label = `${detection.employee_name} (${(detection.similarity * 100).toFixed(1)}%)`
          const textWidth = ctx.measureText(label).width
          
          // Draw label background using mirrored X coordinate
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
          
          // Draw label background using mirrored X coordinate
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

