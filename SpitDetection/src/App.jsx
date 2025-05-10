import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('dashboard')
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [detections, setDetections] = useState([])
  const [detectionData, setDetectionData] = useState([
    { id: 1, location: 'Camera 1', timestamp: '2023-09-01 09:30:45', status: 'Detected' },
    { id: 2, location: 'Camera 2', timestamp: '2023-09-01 10:15:22', status: 'Detected' },
    { id: 3, location: 'Camera 3', timestamp: '2023-09-01 11:05:17', status: 'Clean' },
  ])
  
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  
  const handleLogin = (e) => {
    e.preventDefault()
    // Simple mock authentication - in a real app, this would validate against an API
    if (username === 'admin' && password === 'admin123') {
      setIsLoggedIn(true)
      setError('')
    } else {
      setError('Invalid username or password')
    }
  }

  const handleLogout = () => {
    setIsLoggedIn(false)
    setUsername('')
    setPassword('')
    closeCamera()
  }
  
  const openCamera = async () => {
    setCameraError('')
    
    try {
      console.log("Attempting to access camera...")
      
      // First check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API is not supported in your browser")
      }
      
      // Request camera access with very basic constraints first
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true,
        audio: false
      })
      
      console.log("Camera access granted")
      
      if (videoRef.current) {
        console.log("Setting video source")
        videoRef.current.srcObject = stream
        
        // Make sure video is set to play when ready
        videoRef.current.onloadedmetadata = () => {
          console.log("Video metadata loaded")
          videoRef.current.play()
            .then(() => console.log("Video playing"))
            .catch(e => console.error("Error playing video:", e))
        }
        
        streamRef.current = stream
        setIsCameraOpen(true)
      } else {
        console.error("Video ref is null")
        throw new Error("Video element not available")
      }
    } catch (err) {
      console.error("Error opening camera:", err)
      setCameraError(`Unable to access camera: ${err.message}. Please check permissions.`)
    }
  }
  
  const closeCamera = () => {
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks()
      tracks.forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCameraOpen(false)
  }
  
  const captureDetection = () => {
    // In a real implementation, this would use ML/AI to detect spitting
    // For demo purposes, we'll simulate a detection
    const canvas = document.createElement('canvas')
    const video = videoRef.current
    
    if (video) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      // Convert to base64 for storage/display
      const imageData = canvas.toDataURL('image/jpeg')
      
      const newDetection = {
        id: detections.length + 4, // +4 because we have 3 mock entries
        location: 'Live Camera',
        timestamp: new Date().toLocaleString(),
        status: 'Detected',
        image: imageData
      }
      
      setDetections(prevDetections => [newDetection, ...prevDetections])
      
      // Update the detection data shown in dashboard
      setDetectionData(prevData => [newDetection, ...prevData.slice(0, 9)])
      
      alert("Spit detected and recorded!")
    }
  }
  
  // Simulate random detections (for demo purposes)
  const simulateRandomDetection = () => {
    // 25% chance to detect spitting
    if (Math.random() < 0.25 && isCameraOpen) {
      captureDetection()
    }
  }
  
  useEffect(() => {
    let interval
    if (isCameraOpen) {
      // Check every 5 seconds
      interval = setInterval(simulateRandomDetection, 5000)
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isCameraOpen])

  // Login Page Component
  const LoginPage = () => (
    <div className="login-container">
      <div className="login-form">
        <h1>Spit Detection System</h1>
        <h2>Admin Login</h2>
        {error && <div className="error-message">{error}</div>}
        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="login-button">Login</button>
        </form>
        
        <div className="camera-section">
          <h3>Quick Spit Detection</h3>
          <p>Open camera to monitor and detect in real-time</p>
          
          <div className="camera-container">
            {isCameraOpen ? (
              <>
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline
                  muted
                  className="camera-preview"
                />
                <div className="camera-controls">
                  <button onClick={closeCamera} className="camera-button close">
                    Close Camera
                  </button>
                  <button onClick={captureDetection} className="camera-button capture">
                    Simulate Detection
                  </button>
                </div>
              </>
            ) : (
              <>
                <button onClick={openCamera} className="camera-button open">
                  Open Camera
                </button>
                {cameraError && <div className="camera-error">{cameraError}</div>}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )

  // Dashboard Component
  const Dashboard = () => (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Spit Detection System - Admin Dashboard</h1>
        <button onClick={handleLogout} className="logout-button">Logout</button>
      </div>
      <div className="dashboard-tabs">
        <button 
          className={activeTab === 'dashboard' ? 'active' : ''} 
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
        <button 
          className={activeTab === 'detections' ? 'active' : ''} 
          onClick={() => setActiveTab('detections')}
        >
          Detections
        </button>
        <button 
          className={activeTab === 'cameras' ? 'active' : ''} 
          onClick={() => setActiveTab('cameras')}
        >
          Cameras
        </button>
        <button 
          className={activeTab === 'settings' ? 'active' : ''} 
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
      </div>
      <div className="dashboard-content">
        {activeTab === 'dashboard' && (
          <div className="dashboard-overview">
            <h2>System Overview</h2>
            <div className="stats-cards">
              <div className="stat-card">
                <h3>Total Detections</h3>
                <p className="stat-value">{detectionData.length}</p>
              </div>
              <div className="stat-card">
                <h3>Active Cameras</h3>
                <p className="stat-value">{isCameraOpen ? 1 : 0}</p>
              </div>
              <div className="stat-card">
                <h3>Today's Alerts</h3>
                <p className="stat-value">{detections.length}</p>
              </div>
              <div className="stat-card">
                <h3>System Status</h3>
                <p className="stat-value status-active">Active</p>
              </div>
            </div>
            
            <div className="camera-dashboard">
              <h3>Live Camera Feed</h3>
              <div className="camera-container dashboard-camera">
                {isCameraOpen ? (
                  <>
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline
                      muted
                      className="camera-preview"
                    />
                    <div className="camera-controls">
                      <button onClick={closeCamera} className="camera-button close">
                        Close Camera
                      </button>
                      <button onClick={captureDetection} className="camera-button capture">
                        Capture Detection
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <button onClick={openCamera} className="camera-button open">
                      Open Camera
                    </button>
                    {cameraError && <div className="camera-error">{cameraError}</div>}
                  </>
                )}
              </div>
            </div>
            
            <div className="recent-detections">
              <h3>Recent Detections</h3>
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Location</th>
                    <th>Timestamp</th>
                    <th>Status</th>
                    <th>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {detectionData.map(detection => (
                    <tr key={detection.id}>
                      <td>{detection.id}</td>
                      <td>{detection.location}</td>
                      <td>{detection.timestamp}</td>
                      <td className={detection.status === 'Detected' ? 'status-alert' : 'status-clean'}>
                        {detection.status}
                      </td>
                      <td>
                        {detection.image && (
                          <img 
                            src={detection.image} 
                            alt="Detection evidence" 
                            className="evidence-thumbnail"
                            onClick={() => window.open(detection.image, '_blank')}
                          />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {activeTab === 'detections' && (
          <div className="detections-panel">
            <h2>Detection History</h2>
            <div className="detections-gallery">
              {detectionData.filter(d => d.status === 'Detected').map(detection => (
                <div key={detection.id} className="detection-card">
                  <div className="detection-info">
                    <p><strong>ID:</strong> {detection.id}</p>
                    <p><strong>Location:</strong> {detection.location}</p>
                    <p><strong>Time:</strong> {detection.timestamp}</p>
                  </div>
                  {detection.image && (
                    <img 
                      src={detection.image} 
                      alt="Detection evidence" 
                      className="detection-image"
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        {activeTab === 'cameras' && (
          <div className="cameras-panel">
            <h2>Camera Management</h2>
            <p>Configure and monitor connected cameras here.</p>
          </div>
        )}
        {activeTab === 'settings' && (
          <div className="settings-panel">
            <h2>System Settings</h2>
            <p>Configure system parameters and notification settings.</p>
          </div>
        )}
      </div>
    </div>
  )

  return (
    <div className="app">
      {isLoggedIn ? <Dashboard /> : <LoginPage />}
    </div>
  )
}

export default App
