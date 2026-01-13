import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import './RegisterEmployee.css'

// Backend API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function RegisterEmployee({ userRole }) {
  const navigate = useNavigate()
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const [capturedImages, setCapturedImages] = useState([]) // Store multiple angles
  
  // Set default role based on user's role
  const defaultRole = userRole === 'hr_admin' ? 'employee' : 'employee'
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    role: defaultRole,
  })

  const [errors, setErrors] = useState({})
  const [isCameraModalOpen, setIsCameraModalOpen] = useState(false)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [isCameraLoading, setIsCameraLoading] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const cameraRetryCountRef = useRef(0)
  const MAX_RETRIES = 3
  
  // Multi-angle capture state
  const [currentAngleIndex, setCurrentAngleIndex] = useState(0)
  const [capturedAngles, setCapturedAngles] = useState([])
  const [isCheckingQuality, setIsCheckingQuality] = useState(false)
  const [qualityError, setQualityError] = useState(null)
  const [isProcessingEmbeddings, setIsProcessingEmbeddings] = useState(false)
  const [faceEmbeddings, setFaceEmbeddings] = useState(null)
  const [uploadedImages, setUploadedImages] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [currentUploadIndex, setCurrentUploadIndex] = useState(0)
  const [uploadQualityStatus, setUploadQualityStatus] = useState({}) // Track quality status per index
  const fileInputRef = useRef(null)
  
  const angles = [
    { name: 'Front', instruction: 'Look straight at the camera' },
    { name: 'Slight Left', instruction: 'Turn your head slightly to the left' },
    { name: 'Slight Right', instruction: 'Turn your head slightly to the right' },
    { name: 'Look Up', instruction: 'Look slightly upwards' },
    { name: 'Look Down', instruction: 'Look slightly downwards' },
  ]

  // Check if user is super_admin or hr_admin
  useEffect(() => {
    if (userRole !== 'super_admin' && userRole !== 'hr_admin') {
      navigate('/dashboard')
    }
  }, [userRole, navigate])

  const openCameraModal = () => {
    setIsCameraModalOpen(true)
    setCurrentAngleIndex(0)
    setCapturedAngles([])
    setCameraError(null)
    setIsCameraLoading(true)
    setQualityError(null)
    setFaceEmbeddings(null)
  }

  const closeCameraModal = (preserveCapturedAngles = false) => {
    stopCamera()
    setIsCameraModalOpen(false)
    setCurrentAngleIndex(0)
    if (!preserveCapturedAngles) {
      setCapturedAngles([])
    }
    setCameraError(null)
    setIsCameraLoading(false)
    setQualityError(null)
    setIsCheckingQuality(false)
    setIsProcessingEmbeddings(false)
  }

  const startCamera = async (retryCount = 0) => {
    try {
      setIsCameraLoading(true)
      setCameraError(null)

      // Wait for video element to exist (important!)
      if (!videoRef.current) {
        console.log('⏳ Waiting for video element to mount...')
        setTimeout(() => {
          if (retryCount < MAX_RETRIES) {
            startCamera(retryCount + 1)
          } else {
            setCameraError('Video element not found. Please refresh the page.')
            setIsCameraLoading(false)
          }
        }, 300)
        return
      }

      // Stop old stream if any
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
        streamRef.current = null
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      })

      // Video might get unmounted if modal closes mid-request
      if (!videoRef.current) {
        mediaStream.getTracks().forEach(t => t.stop())
        throw new Error('Video element was removed')
      }

      videoRef.current.srcObject = mediaStream
      streamRef.current = mediaStream

      // Wait for metadata (ensures videoWidth/videoHeight available)
      await new Promise((resolve, reject) => {
        const video = videoRef.current
        if (!video) return reject(new Error('Video element not available'))

        const onLoaded = () => {
          video.removeEventListener('loadedmetadata', onLoaded)
          video.removeEventListener('error', onError)
          resolve()
        }

        const onError = () => {
          video.removeEventListener('loadedmetadata', onLoaded)
          video.removeEventListener('error', onError)
          reject(new Error('Video failed to load stream'))
        }

        if (video.readyState >= 1) {
          onLoaded()
        } else {
          video.addEventListener('loadedmetadata', onLoaded)
          video.addEventListener('error', onError)
          setTimeout(() => reject(new Error('Video load timeout')), 5000)
        }
      })

      setIsCameraActive(true)
      setIsCameraLoading(false)
      cameraRetryCountRef.current = 0
    } catch (error) {
      console.error('❌ Error accessing camera:', error)
      cameraRetryCountRef.current = retryCount + 1

      let msg = 'Unable to access camera. '
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        msg += 'Please allow camera access in your browser settings.'
        setCameraError(msg)
        setIsCameraLoading(false)
        setIsCameraActive(false)
        return
      }
      if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        msg += 'No camera found on your device.'
        setCameraError(msg)
        setIsCameraLoading(false)
        setIsCameraActive(false)
        return
      }
      if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
        msg += 'Camera is being used by another application.'
        setCameraError(msg)
        setIsCameraLoading(false)
        setIsCameraActive(false)
        return
      }

      if (retryCount < MAX_RETRIES) {
        console.log(`🔄 Retrying camera access (${retryCount + 1}/${MAX_RETRIES})...`)
        setTimeout(() => startCamera(retryCount + 1), 1000 * (retryCount + 1))
        return
      }

      msg += 'Please refresh the page and try again.'
      setCameraError(msg)
      setIsCameraLoading(false)
      setIsCameraActive(false)
    }
  }

  // Start camera when modal opens
  useEffect(() => {
    if (isCameraModalOpen) {
      startCamera(0)
    } else {
      stopCamera()
    }

    return () => stopCamera()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCameraModalOpen])

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setIsCameraActive(false)
    setIsCameraLoading(false)
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  const checkImageQuality = async (imageData) => {
    try {
      setIsCheckingQuality(true)
      setQualityError(null)

      // Get auth token
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        throw new Error('Not authenticated')
      }

      const response = await fetch(`${API_BASE_URL}/api/face/check-quality`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`
        },
        body: JSON.stringify({ image: imageData })
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Quality check failed')
      }

      return result
    } catch (error) {
      console.error('Quality check error:', error)
      return {
        success: false,
        quality_ok: false,
        message: error.message || 'Failed to check image quality'
      }
    } finally {
      setIsCheckingQuality(false)
    }
  }

  const capturePhoto = async () => {
    // Prevent multiple simultaneous captures
    if (isCheckingQuality || isProcessingEmbeddings) {
      console.log('Capture blocked: isCheckingQuality=', isCheckingQuality, 'isProcessingEmbeddings=', isProcessingEmbeddings)
      return
    }

    console.log('Capture button clicked, currentAngleIndex:', currentAngleIndex, 'qualityError:', qualityError)

    if (videoRef.current && canvasRef.current) {
      // Clear any previous error when attempting to capture
      setQualityError(null)
      console.log('Quality error cleared, starting capture...')
      
      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext('2d')

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      context.drawImage(video, 0, 0)

      const imageData = canvas.toDataURL('image/png')
      const currentAngle = angles[currentAngleIndex]

      // Check image quality first
      const qualityResult = await checkImageQuality(imageData)

      if (!qualityResult.quality_ok) {
        console.log('Quality check failed:', qualityResult.message)
        setQualityError(qualityResult.message || 'Image quality check failed')
        // Don't proceed - allow user to retry by clicking again
        // Button remains enabled so user can retry
        return
      }

      console.log('Quality check passed, proceeding with capture')

      // Quality is good, proceed with capture
      setQualityError(null)
      
      // Add to captured angles
      const newCapturedAngles = [
        ...capturedAngles,
        {
          angle: currentAngle.name,
          image: imageData,
          index: currentAngleIndex
        }
      ]
      setCapturedAngles(newCapturedAngles)
      
      // Move to next angle or finish
      if (currentAngleIndex < angles.length - 1) {
        setCurrentAngleIndex(currentAngleIndex + 1)
      } else {
        // All angles captured - process embeddings
        await processAllImages(newCapturedAngles)
      }
    }
  }

  const processAllImages = async (capturedAnglesData) => {
    try {
      setIsProcessingEmbeddings(true)
      setQualityError(null)

      // Get auth token
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        throw new Error('Not authenticated')
      }

      // Prepare images for backend (extract base64 data)
      const imagesForBackend = capturedAnglesData.map(item => ({
        angle: item.angle,
        image: item.image // Already base64 data URL
      }))

      const response = await fetch(`${API_BASE_URL}/api/face/process-embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`
        },
        body: JSON.stringify({ images: imagesForBackend })
      })

      const result = await response.json()

      if (!response.ok || !result.success) {
        throw new Error(result.message || 'Failed to process embeddings')
      }

      // Store embeddings
      setFaceEmbeddings(result)
      setCapturedImages(capturedAnglesData.map(a => a.image))
      // Keep capturedAngles state so it displays after modal closes
      setCapturedAngles(capturedAnglesData)
      
      // If processing from upload (not camera), store in uploadedImages
      if (isUploadModalOpen) {
        setUploadedImages(capturedAnglesData)
        // Close upload modal but preserve uploaded images
        setTimeout(() => {
          closeUploadModal(true) // Pass true to preserve uploaded images
        }, 500)
      }
      
      // Close camera modal but preserve captured angles (only if camera modal is open)
      if (isCameraModalOpen) {
      stopCamera()
      setTimeout(() => {
        closeCameraModal(true) // Pass true to preserve captured angles
      }, 500)
      }

      console.log('Embeddings processed successfully:', result)
      
    } catch (error) {
      console.error('Processing error:', error)
      setQualityError(error.message || 'Failed to process face embeddings')
    } finally {
      setIsProcessingEmbeddings(false)
    }
  }

  const retakeCurrentAngle = () => {
    // Remove the last captured angle and stay on current index
    const newCapturedAngles = capturedAngles.filter(
      a => a.index !== currentAngleIndex - 1
    )
    setCapturedAngles(newCapturedAngles)
    setCurrentAngleIndex(Math.max(0, currentAngleIndex - 1))
  }

  const retakeAll = () => {
    setCapturedAngles([])
    setCurrentAngleIndex(0)
    setCapturedImages([])
    setUploadedImages([])
    setFaceEmbeddings(null)
    if (!isCameraActive) {
      startCamera()
    }
  }


  const openUploadModal = () => {
    setIsUploadModalOpen(true)
    setCurrentUploadIndex(0)
    setUploadedImages([])
    setUploadQualityStatus({})
    setQualityError(null)
    setFaceEmbeddings(null)
  }

  const closeUploadModal = (preserveUploadedImages = false) => {
    setIsUploadModalOpen(false)
    setCurrentUploadIndex(0)
    if (!preserveUploadedImages) {
      setUploadedImages([])
      setUploadQualityStatus({})
    }
    setQualityError(null)
    setIsUploading(false)
    setIsCheckingQuality(false)
    setIsProcessingEmbeddings(false)
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleSingleFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if (!validTypes.includes(file.type)) {
      setQualityError('Please upload a valid image file (JPEG, PNG, or WebP)')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
      return
    }

    setIsUploading(true)
    setQualityError(null)

    try {
      // Convert file to base64
      const imageData = await new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => resolve(e.target.result)
        reader.onerror = reject
        reader.readAsDataURL(file)
      })

      // Check image quality
      setIsCheckingQuality(true)
      const qualityResult = await checkImageQuality(imageData)

      if (!qualityResult.quality_ok) {
        setQualityError(qualityResult.message || 'Image quality check failed. Please upload a clear, well-lit face photo.')
        setUploadQualityStatus(prev => ({
          ...prev,
          [currentUploadIndex]: { status: 'failed', message: qualityResult.message }
        }))
        setIsUploading(false)
        setIsCheckingQuality(false)
        // Reset file input to allow retry
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
        return
      }

      // Quality passed - add to uploaded images
      const currentAngle = angles[currentUploadIndex]
      const newUploadedImage = {
        angle: currentAngle?.name || `Image ${currentUploadIndex + 1}`,
        image: imageData,
        index: currentUploadIndex
      }

      const newUploadedImages = [...uploadedImages]
      // Replace if already exists at this index, otherwise add
      const existingIndex = newUploadedImages.findIndex(img => img.index === currentUploadIndex)
      if (existingIndex >= 0) {
        newUploadedImages[existingIndex] = newUploadedImage
      } else {
        newUploadedImages.push(newUploadedImage)
      }
      // Sort by index
      newUploadedImages.sort((a, b) => a.index - b.index)
      setUploadedImages(newUploadedImages)

      // Update quality status
      setUploadQualityStatus(prev => ({
        ...prev,
        [currentUploadIndex]: { status: 'passed' }
      }))

      setQualityError(null)

      // Check if all images are uploaded
      if (newUploadedImages.length === angles.length) {
        // All images uploaded - process embeddings
        await processAllImages(newUploadedImages)
      } else {
        // Move to next angle
        const nextIndex = Math.min(currentUploadIndex + 1, angles.length - 1)
        setCurrentUploadIndex(nextIndex)
      }

    } catch (error) {
      console.error('Upload error:', error)
      setQualityError(error.message || 'Failed to upload image. Please try again.')
      setUploadQualityStatus(prev => ({
        ...prev,
        [currentUploadIndex]: { status: 'error', message: error.message }
      }))
    } finally {
      setIsUploading(false)
      setIsCheckingQuality(false)
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const deleteUploadedImage = (index) => {
    setUploadedImages(prev => prev.filter(img => img.index !== index))
    setUploadQualityStatus(prev => {
      const newStatus = { ...prev }
      delete newStatus[index]
      return newStatus
    })
    // If we delete the current index or before it, go back
    if (index <= currentUploadIndex) {
      setCurrentUploadIndex(Math.max(0, index))
    }
  }

  const retryUploadAtIndex = (index) => {
    setCurrentUploadIndex(index)
    setQualityError(null)
    setUploadQualityStatus(prev => {
      const newStatus = { ...prev }
      delete newStatus[index]
      return newStatus
    })
    // Trigger file input
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    // Clear error for this field
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }))
    }
  }

  const validateForm = () => {
    const newErrors = {}

    if (!formData.name.trim()) {
      newErrors.name = 'Employee name is required'
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required'
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address'
    }

    if (!formData.role) {
      newErrors.role = 'Please select a role'
    }

    if (capturedImages.length === 0 && uploadedImages.length === 0) {
      newErrors.photo = 'Please capture or upload employee photos from all angles'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateForm()) {
      return
    }

    // Check if face embeddings are available
    if (!faceEmbeddings || !faceEmbeddings.average_embedding) {
      setErrors(prev => ({
        ...prev,
        photo: 'Please capture employee photos from all angles'
      }))
      return
    }

    try {
      // Get auth token
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        setErrors(prev => ({
          ...prev,
          submit: 'Not authenticated. Please log in again.'
        }))
        return
      }

      // Call the edge function using Supabase client (handles CORS automatically)
      const { data: result, error: functionError } = await supabase.functions.invoke(
        'register-employee',
        {
          body: {
            name: formData.name.trim(),
            email: formData.email.trim(),
            role: formData.role,
            face_embeddings: faceEmbeddings.average_embedding // Base64 encoded embedding
          }
        }
      )

      if (functionError) {
        throw new Error(functionError.message || 'Failed to register employee')
      }

      if (!result || !result.success) {
        throw new Error(result?.error || 'Failed to register employee')
      }

      // Success!
      const successMessage = result.user?.requires_login 
        ? `${result.user.role === 'hr_admin' ? 'HR Admin' : result.user.role === 'super_admin' ? 'Super Admin' : 'Employee'} "${result.user.name}" registered successfully! ${result.user.invitation_sent ? 'Invitation email sent.' : ''}`
        : `Employee "${result.user.name}" registered successfully!`
      
      alert(successMessage)
      
      // Reset form
      setFormData({
        name: '',
        email: '',
        role: defaultRole,
      })
      setCapturedImages([])
      setCapturedAngles([])
      setUploadedImages([])
      setFaceEmbeddings(null)
      setErrors({})
      
      // Optionally navigate to dashboard or employee list
      // navigate('/dashboard')
      
    } catch (error) {
      console.error('Registration error:', error)
      setErrors(prev => ({
        ...prev,
        submit: error.message || 'Failed to register employee. Please try again.'
      }))
    }
  }


  // Check access
  if (userRole !== 'super_admin' && userRole !== 'hr_admin') {
    return null
  }

  const isSuperAdmin = userRole === 'super_admin'
  const isHrAdmin = userRole === 'hr_admin'
  
  // Ensure HR Admin can only set role to employee
  useEffect(() => {
    if (isHrAdmin && formData.role !== 'employee') {
      setFormData(prev => ({ ...prev, role: 'employee' }))
    }
  }, [isHrAdmin, formData.role])

  return (
    <div className="register-employee-container">
      <div className="register-employee-header">
        <button 
          onClick={() => navigate('/dashboard')}
          className="back-button"
        >
          ← Back to Dashboard
        </button>
        <h1>Register New Employee</h1>
        <p>
          {isSuperAdmin 
            ? 'Add a new employee to the system with face recognition'
            : 'Add a new employee to the system (HR Admin can only register employees)'
          }
        </p>
      </div>

      <div className="register-employee-content">
        <form onSubmit={handleSubmit} className="register-form">
          <div className="form-section">
            <h2>Employee Information</h2>
            
            <div className="form-group">
              <label htmlFor="name">
                Employee Name <span className="required">*</span>
              </label>
              <input
                id="name"
                name="name"
                type="text"
                value={formData.name}
                onChange={handleChange}
                placeholder="Enter employee full name"
                className={errors.name ? 'error' : ''}
              />
              {errors.name && <span className="error-message">{errors.name}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="email">
                Email Address <span className="required">*</span>
              </label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="employee@company.com"
                className={errors.email ? 'error' : ''}
              />
              {errors.email && <span className="error-message">{errors.email}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="role">
                Access Level <span className="required">*</span>
                {isHrAdmin && (
                  <span className="role-restriction"> (HR Admin can only register employees)</span>
                )}
              </label>
              {isSuperAdmin ? (
                <>
                  <select
                    id="role"
                    name="role"
                    value={formData.role}
                    onChange={handleChange}
                    className={errors.role ? 'error' : ''}
                  >
                    <option value="employee">Employee</option>
                    <option value="hr_admin">HR Administrator</option>
                    <option value="super_admin">Super Administrator</option>
                  </select>
                  <p className="field-hint">
                    Select the access level for this employee
                  </p>
                </>
              ) : (
                <>
                  <select
                    id="role"
                    name="role"
                    value="employee"
                    disabled
                    className="disabled-select"
                  >
                    <option value="employee">Employee</option>
                  </select>
                  <p className="field-hint">
                    <span className="restriction-note">⚠️ HR Administrators can only register employees</span>
                  </p>
                </>
              )}
              {errors.role && <span className="error-message">{errors.role}</span>}
            </div>
          </div>

          <div className="form-section">
            <h2>Face Registration</h2>
            <p className="section-description">
              Capture employee's face for attendance recognition system
            </p>

            <div className="camera-section">
              {(capturedImages.length === 0 && uploadedImages.length === 0) ? (
                <div className="camera-container">
                  <div className="camera-placeholder">
                    <div className="camera-icon">📷</div>
                    <p>Capture or upload face images from multiple angles for better recognition</p>
                    <div className="face-input-options">
                    <button
                      type="button"
                      onClick={openCameraModal}
                      className="camera-button"
                    >
                        📷 Capture from Camera
                    </button>
                      <span className="option-divider">OR</span>
                      <button
                        type="button"
                        onClick={openUploadModal}
                        className="upload-button"
                      >
                        📁 Upload Images
                      </button>
                    </div>
                    <p className="camera-hint">
                      Capture or upload 5 angles: Front, Left, Right, Up, Down
                    </p>
                    {qualityError && (
                      <div className="error-message photo-error">{qualityError}</div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="captured-photos-container">
                  <div className="captured-photos-grid">
                    {(capturedAngles.length > 0 ? capturedAngles : uploadedImages).map((item, idx) => (
                      <div key={idx} className="captured-photo-item">
                        <img 
                          src={item.image} 
                          alt={`${item.angle} angle`} 
                          className="captured-photo-thumb"
                        />
                        <p className="photo-angle-label">{item.angle}</p>
                      </div>
                    ))}
                  </div>
                  <div className="photo-controls">
                    <button
                      type="button"
                      onClick={retakeAll}
                      className="button-secondary"
                    >
                      {capturedAngles.length > 0 ? 'Retake All Photos' : 'Remove All Photos'}
                    </button>
                    <p className="photo-status">
                      ✓ {(capturedAngles.length > 0 ? capturedAngles : uploadedImages).length} angles {capturedAngles.length > 0 ? 'captured' : uploadedImages.length > 0 ? 'uploaded' : ''} successfully
                    </p>
                  </div>
                </div>
              )}
              {errors.photo && (
                <span className="error-message photo-error">{errors.photo}</span>
              )}
              {errors.submit && (
                <span className="error-message photo-error">{errors.submit}</span>
              )}
            </div>
          </div>

          <div className="form-actions">
            <button
              type="button"
              onClick={() => navigate('/dashboard')}
              className="button-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="button-primary submit-button"
            >
              Register Employee
            </button>
          </div>
        </form>
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Camera Modal */}
      {isCameraModalOpen && (
        <div className="camera-modal-overlay" onClick={(e) => {
          if (e.target === e.currentTarget) {
            closeCameraModal()
          }
        }}>
          <div className="camera-modal-content">
            <div className="camera-modal-header">
              <h2>Capture Face - Multi-Angle</h2>
              <button 
                className="modal-close-button"
                onClick={closeCameraModal}
                aria-label="Close"
              >
                ×
              </button>
            </div>
            
            <div className="camera-modal-body">
              {/* Always render video element when modal is open */}
              <div className="video-wrapper">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="camera-video"
                />
                <div className="face-guide"></div>

                {/* Loading Overlay */}
                {(isCameraLoading || !isCameraActive) && !cameraError && (
                  <div className="camera-loading-overlay">
                    <div className="loading-spinner"></div>
                    <p>Starting camera...</p>
                    <p className="loading-hint">Please ensure camera permissions are granted</p>
                  </div>
                )}

                {/* Error Overlay */}
                {cameraError && (
                  <div className="camera-error-overlay">
                    <div className="error-icon">⚠️</div>
                    <h3>Camera Error</h3>
                    <p>{cameraError}</p>
                    <div className="error-actions">
                      <button
                        type="button"
                        onClick={() => {
                          setCameraError(null)
                          setIsCameraLoading(true)
                          cameraRetryCountRef.current = 0
                          startCamera(0)
                        }}
                        className="button-primary"
                      >
                        Try Again
                      </button>
                      <button
                        type="button"
                        onClick={closeCameraModal}
                        className="button-secondary"
                      >
                        Close
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Angle instructions and controls - only show when camera is ready */}
              {!cameraError && !isCameraLoading && isCameraActive && (
                <>
                  <div className="angle-progress">
                    <div className="angle-progress-bar">
                      <div 
                        className="angle-progress-fill"
                        style={{ width: `${((currentAngleIndex + 1) / angles.length) * 100}%` }}
                      ></div>
                    </div>
                    <p className="angle-progress-text">
                      Angle {currentAngleIndex + 1} of {angles.length}
                    </p>
                  </div>
                  
                  <div className="angle-instruction">
                    <h3>{angles[currentAngleIndex].name}</h3>
                    <p>{angles[currentAngleIndex].instruction}</p>
                  </div>

                  {/* Quality Error Message */}
                  {qualityError && (
                    <div className="quality-error-message">
                      <div className="error-icon-small">⚠️</div>
                      <p>{qualityError}</p>
                      <p className="error-hint">Please adjust your position and try again</p>
                      <button
                        type="button"
                        onClick={capturePhoto}
                        className="button-primary retry-error-button"
                        disabled={isCheckingQuality || isProcessingEmbeddings}
                      >
                        {isCheckingQuality ? 'Checking Quality...' : '📸 Capture Again'}
                      </button>
                    </div>
                  )}

                  {/* Processing Embeddings Message */}
                  {isProcessingEmbeddings && (
                    <div className="processing-message">
                      <div className="loading-spinner-small"></div>
                      <p>Processing face embeddings...</p>
                    </div>
                  )}
                  
                  {/* Hide preview when error is shown to save space */}
                  {!qualityError && capturedAngles.length > 0 && (
                    <div className="captured-angles-preview">
                      {capturedAngles.map((item, idx) => (
                        <div key={idx} className="angle-preview-item">
                          <img src={item.image} alt={item.angle} />
                          <span className="angle-preview-label">{item.angle}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <div className="camera-modal-controls">
                    <button
                      type="button"
                      onClick={retakeCurrentAngle}
                      className="button-secondary"
                      disabled={currentAngleIndex === 0 && capturedAngles.length === 0}
                    >
                      ← Previous
                    </button>
                    <button
                      type="button"
                      onClick={capturePhoto}
                      className="button-primary capture-button"
                      disabled={isCheckingQuality || isProcessingEmbeddings || qualityError}
                    >
                      {isCheckingQuality 
                        ? 'Checking Quality...'
                        : isProcessingEmbeddings
                        ? 'Processing...'
                        : currentAngleIndex < angles.length - 1 
                        ? `📸 Capture ${angles[currentAngleIndex].name}`
                        : `📸 Capture ${angles[currentAngleIndex].name} & Finish`
                      }
                    </button>
                    <button
                      type="button"
                      onClick={closeCameraModal}
                      className="button-secondary"
                    >
                      Cancel
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {isUploadModalOpen && (
        <div className="camera-modal-overlay" onClick={(e) => {
          if (e.target === e.currentTarget) {
            closeUploadModal()
          }
        }}>
          <div className="camera-modal-content">
            <div className="camera-modal-header">
              <h2>Upload Face Images - Multi-Angle</h2>
              <button 
                className="modal-close-button"
                onClick={() => closeUploadModal()}
                aria-label="Close"
              >
                ×
              </button>
            </div>
            
            <div className="camera-modal-body">
              <div className="upload-instructions">
                <div className="angle-progress">
                  <div className="angle-progress-bar">
                    <div 
                      className="angle-progress-fill"
                      style={{ width: `${((uploadedImages.length) / angles.length) * 100}%` }}
                    ></div>
                  </div>
                  <p className="angle-progress-text">
                    {uploadedImages.length} of {angles.length} images uploaded
                  </p>
                </div>
                
                <div className="angle-instruction">
                  <h3>{angles[currentUploadIndex]?.name || 'Upload Image'}</h3>
                  <p>{angles[currentUploadIndex]?.instruction || 'Upload a clear face photo'}</p>
                </div>

                {/* Quality Error Message */}
                {qualityError && (
                  <div className="quality-error-message">
                    <div className="error-icon-small">⚠️</div>
                    <p>{qualityError}</p>
                    <p className="error-hint">Please upload a different image</p>
                    <button
                      type="button"
                      onClick={() => {
                        setQualityError(null)
                        if (fileInputRef.current) {
                          fileInputRef.current.click()
                        }
                      }}
                      className="button-primary retry-error-button"
                      disabled={isCheckingQuality || isUploading}
                    >
                      {isCheckingQuality ? 'Checking Quality...' : '📁 Upload Again'}
                    </button>
                  </div>
                )}

                {/* Processing Embeddings Message */}
                {isProcessingEmbeddings && (
                  <div className="processing-message">
                    <div className="loading-spinner-small"></div>
                    <p>Processing face embeddings...</p>
                  </div>
                )}

                {/* Uploaded Images Preview */}
                {uploadedImages.length > 0 && (
                  <div className="uploaded-images-preview">
                    <h4>Uploaded Images:</h4>
                    <div className="uploaded-images-grid">
                      {angles.map((angle, idx) => {
                        const uploadedImage = uploadedImages.find(img => img.index === idx)
                        const qualityStatus = uploadQualityStatus[idx]
                        
                        return (
                          <div key={idx} className={`uploaded-image-item ${uploadedImage ? 'uploaded' : 'pending'} ${qualityStatus?.status === 'failed' ? 'failed' : ''}`}>
                            {uploadedImage ? (
                              <>
                                <img src={uploadedImage.image} alt={angle.name} />
                                <div className="uploaded-image-overlay">
                                  {qualityStatus?.status === 'passed' && (
                                    <span className="quality-badge success">✓</span>
                                  )}
                                  {qualityStatus?.status === 'failed' && (
                                    <span className="quality-badge error">✗</span>
                                  )}
                                  <button
                                    type="button"
                                    className="delete-image-btn"
                                    onClick={() => deleteUploadedImage(idx)}
                                    title="Delete this image"
                                  >
                                    🗑️
                                  </button>
                                </div>
                                <p className="uploaded-image-label">{angle.name}</p>
                                {qualityStatus?.status === 'failed' && (
                                  <button
                                    type="button"
                                    className="retry-upload-btn"
                                    onClick={() => retryUploadAtIndex(idx)}
                                  >
                                    Retry Upload
                                  </button>
                                )}
                              </>
                            ) : (
                              <>
                                <div className="upload-placeholder">
                                  <span>{idx + 1}</span>
                                </div>
                                <p className="uploaded-image-label">{angle.name}</p>
                              </>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
              
              <div className="camera-modal-controls">
                <button
                  type="button"
                  onClick={() => {
                    if (currentUploadIndex > 0) {
                      setCurrentUploadIndex(currentUploadIndex - 1)
                      setQualityError(null)
                    }
                  }}
                  className="button-secondary"
                  disabled={currentUploadIndex === 0 || isCheckingQuality || isUploading || isProcessingEmbeddings}
                >
                  ← Previous
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (fileInputRef.current) {
                      fileInputRef.current.click()
                    }
                  }}
                  className="button-primary capture-button"
                  disabled={isCheckingQuality || isUploading || isProcessingEmbeddings}
                >
                  {isCheckingQuality 
                    ? 'Checking Quality...'
                    : isUploading
                    ? 'Uploading...'
                    : isProcessingEmbeddings
                    ? 'Processing...'
                    : currentUploadIndex < angles.length - 1 
                    ? `📁 Upload ${angles[currentUploadIndex]?.name}`
                    : uploadedImages.length === angles.length
                    ? '✓ All Uploaded'
                    : `📁 Upload ${angles[currentUploadIndex]?.name} & Finish`
                  }
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png,image/webp"
                  onChange={handleSingleFileUpload}
                  style={{ display: 'none' }}
                />
                <button
                  type="button"
                  onClick={() => closeUploadModal()}
                  className="button-secondary"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default RegisterEmployee

