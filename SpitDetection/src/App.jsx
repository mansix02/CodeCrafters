import { useState, useRef, useEffect } from 'react'
import './App.css'

// Base URL for the Python camera server
const API_BASE_URL = 'http://localhost:5000';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('dashboard')
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [currentPage, setCurrentPage] = useState('login') // login, dashboard, camera
  const [detections, setDetections] = useState([])
  const [cameraStatus, setCameraStatus] = useState('idle') // idle, loading, ready, error
  const [modelStatus, setModelStatus] = useState({ initialized: false, loading: false })
  const [detectionData, setDetectionData] = useState([
    { id: 1, location: 'Camera 1', timestamp: '2023-09-01 09:30:45', status: 'Spitting Detected', severity: 'high' },
    { id: 2, location: 'Camera 2', timestamp: '2023-09-01 10:15:22', status: 'Spitting Detected', severity: 'medium' },
    { id: 3, location: 'Camera 3', timestamp: '2023-09-01 11:05:17', status: 'Clean', severity: 'none' },
  ])
  
  const imgRef = useRef(null)
  
  const handleLogin = (e) => {
    e.preventDefault()
    // Simple mock authentication - in a real app, this would validate against an API
    if (username === 'admin' && password === 'admin123') {
      setIsLoggedIn(true)
      setCurrentPage('dashboard')
      setError('')
    } else {
      setError('Invalid username or password')
    }
  }

  const handleLogout = () => {
    setIsLoggedIn(false)
    setUsername('')
    setPassword('')
    setCurrentPage('login')
    closeCamera()
  }

  // Check model status
  const checkModelStatus = async () => {
    try {
      setModelStatus(prev => ({ ...prev, loading: true }));
      const response = await fetch(`${API_BASE_URL}/model_status`);
      const data = await response.json();
      setModelStatus({
        initialized: data.initialized,
        loading: false,
        path: data.model_path
      });
    } catch (error) {
      console.error("Error checking model status:", error);
      setModelStatus({
        initialized: false,
        loading: false,
        error: error.message
      });
    }
  };

  // This useEffect initializes the camera when we navigate to the camera page
  useEffect(() => {
    // Check model status when app loads
    checkModelStatus();
    
    // Only try to initialize camera if we're on the camera page
    if (currentPage === 'camera') {
      initCamera()
    }
    
    // Clean up when leaving the camera page
    return () => {
      if (currentPage !== 'camera' && isCameraOpen) {
        closeCamera()
      }
    }
  }, [currentPage])
  
  // Start a periodic check for new detections
  useEffect(() => {
    let checkInterval;
    
    if (isCameraOpen) {
      // Check for new detections every 3 seconds
      checkInterval = setInterval(() => {
        checkForDetections()
      }, 3000)
    }
    
    return () => {
      if (checkInterval) {
        clearInterval(checkInterval)
      }
    }
  }, [isCameraOpen])
  
  const checkForDetections = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/get_last_detection`)
      const data = await response.json()
      
      if (data.success && data.detection) {
        // Add this detection to our list
        const newDetection = {
          id: detections.length + 4,
          location: 'Live Camera',
          timestamp: data.detection.timestamp,
          status: 'Spitting Detected',
          severity: 'high',
          image: data.detection.image
        }
        
        setDetections(prevDetections => {
          // Check if we already have this detection (based on timestamp)
          const exists = prevDetections.some(d => d.timestamp === data.detection.timestamp);
          if (exists) return prevDetections;
          return [newDetection, ...prevDetections];
        });
        
        setDetectionData(prevData => {
          // Check if we already have this detection
          const exists = prevData.some(d => d.timestamp === data.detection.timestamp);
          if (exists) return prevData;
          return [newDetection, ...prevData.slice(0, 9)];
        });
        
        // Play alert sound and show alert
        const audio = new Audio('/alert.mp3');
        audio.play().catch(e => console.log("Audio play error (may be expected):", e));
        
        alert("⚠️ ALERT: Spitting detected and recorded! Authorities have been notified.");
      }
    } catch (error) {
      console.error("Error checking for detections:", error)
    }
  }
  
  const initCamera = async () => {
    setCameraStatus('loading')
    setCameraError('')
    
    try {
      console.log("Starting camera on Python server...")
      
      // Tell the Python server to start the camera
      const response = await fetch(`${API_BASE_URL}/start_camera`, {
        method: 'POST'
      })
      
      if (!response.ok) {
        throw new Error("Failed to start camera on server")
      }
      
      // Set a timeout to give the camera time to initialize
      setTimeout(() => {
        setCameraStatus('ready')
        setIsCameraOpen(true)
        
        // Also check model status
        checkModelStatus();
      }, 1000)
      
    } catch (err) {
      console.error("Camera initialization error:", err)
      setCameraError(`Camera error: ${err.message}. Make sure the Python server is running.`)
      setCameraStatus('error')
    }
  }
  
  const openCamera = () => {
    console.log("Opening camera page...")
    setCurrentPage('camera')
  }
  
  const closeCamera = async () => {
    console.log("Closing camera...")
    
    try {
      // Tell the Python server to stop the camera
      await fetch(`${API_BASE_URL}/stop_camera`, {
        method: 'POST'
      })
    } catch (err) {
      console.error("Error stopping camera:", err)
    }
    
    setIsCameraOpen(false)
    setCameraStatus('idle')
    setCurrentPage(isLoggedIn ? 'dashboard' : 'login')
  }
  
  const captureDetection = async () => {
    if (!isCameraOpen) {
      console.error("Camera not open")
      return
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/capture_detection`)
      const data = await response.json()
      
      if (data.success && data.detection) {
        // Add this detection to our list
        const newDetection = {
          id: detections.length + 4,
          location: 'Live Camera',
          timestamp: data.detection.timestamp,
          status: 'Spitting Detected',
          severity: 'medium',
          image: data.detection.image
        }
        
        setDetections(prevDetections => [newDetection, ...prevDetections])
        setDetectionData(prevData => [newDetection, ...prevData.slice(0, 9)])
        
        alert("Manual detection recorded!")
      } else {
        alert("Failed to capture detection")
      }
    } catch (err) {
      console.error("Error capturing detection:", err)
      alert("Error capturing detection: " + err.message)
    }
  }

  // Camera Page Component
  const CameraPage = () => (
    <div className="camera-page">
      <div className="camera-header">
        <button onClick={closeCamera} className="camera-back-button">
          <span>←</span> Back
        </button>
        <h2>AI Spitting Detection System</h2>
      </div>
      
      <div className="camera-view-container">
        {cameraStatus === 'error' ? (
          <div className="camera-error-full">
            <h3>Camera Error</h3>
            <p>{cameraError}</p>
            <button onClick={initCamera}>Retry Camera</button>
          </div>
        ) : cameraStatus === 'loading' ? (
          <div className="camera-loading-full">
            <div className="loader"></div>
            <p>Initializing camera...</p>
          </div>
        ) : (
          <div className="camera-live-container">
            <div className="camera-live-feed">
              <img 
                src={`${API_BASE_URL}/video_feed?${new Date().getTime()}`} 
                alt="Live Camera Feed" 
                ref={imgRef}
              />
              <div className="camera-controls">
                <div className="camera-status">
                  <span 
                    className={`status-indicator ${
                      modelStatus.initialized ? 'status-green' : 'status-red'
                    }`}
                  ></span>
                  <span className="status-text">
                    AI Model: {modelStatus.initialized ? 'Active' : 'Not Ready'}
                  </span>
                </div>
                
                <div className="camera-status">
                  <span className="status-indicator status-green"></span>
                  <span className="status-text">Camera Active</span>
                </div>
                
                <button onClick={captureDetection} className="capture-button">
                  Manual Capture
                </button>
              </div>
            </div>
            
            <div className="detection-info">
              <h3>Detection Information</h3>
              {detections.length > 0 ? (
                <div className="detection-card">
                  <h4>Latest Detection</h4>
                  <p>Time: {detections[0].timestamp}</p>
                  <p>Status: {detections[0].status}</p>
                  <p>Severity: {detections[0].severity}</p>
                  {detections[0].image && (
                    <img 
                      src={detections[0].image} 
                      alt="Detection" 
                      className="detection-image"
                    />
                  )}
                </div>
              ) : (
                <p>No detections recorded yet. System monitoring in progress.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )

  // Login Page Component  
  const LoginPage = () => (
    <div className="login-container">
      <div className="login-form-container">
        <div className="login-header">
          <h1>Spitting Detection System</h1>
          <p>AI-powered public hygiene monitoring</p>
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleLogin} className="login-form">
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
        
        <div className="login-help">
          <p>Demo credentials: admin / admin123</p>
        </div>
      </div>
      
      <div className="login-info">
        <h2>AI-Powered Spitting Detection</h2>
        <p>
          Our system uses advanced YOLOv8 computer vision technology to detect spitting behavior 
          in public spaces, promoting hygiene and supporting civic enforcement efforts.
        </p>
        <ul>
          <li>Real-time AI detection</li>
          <li>Automatic alerts</li>
          <li>Evidence capture</li>
          <li>Dashboard reporting</li>
        </ul>
      </div>
    </div>
  )

  // Dashboard Component
  const Dashboard = () => (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Spitting Detection Dashboard</h1>
        <div className="user-controls">
          <span>Welcome, {username || 'Admin'}</span>
          <button onClick={handleLogout} className="logout-button">Logout</button>
        </div>
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
          className={activeTab === 'settings' ? 'active' : ''}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
      </div>
      
      <div className="dashboard-content">
        {activeTab === 'dashboard' && (
          <div className="dashboard-main">
            <div className="stats-cards">
              <div className="stat-card">
                <h3>Active Cameras</h3>
                <div className="stat-value">4</div>
                <div className="stat-trend positive">All Online</div>
              </div>
              
              <div className="stat-card">
                <h3>Detections Today</h3>
                <div className="stat-value">{detectionData.length}</div>
                <div className="stat-trend negative">+3 from yesterday</div>
              </div>
              
              <div className="stat-card">
                <h3>Response Rate</h3>
                <div className="stat-value">92%</div>
                <div className="stat-trend positive">+5% from last week</div>
              </div>
              
              <div className="stat-card">
                <h3>AI Model Status</h3>
                <div className="stat-value">
                  {modelStatus.initialized ? 'Active' : 'Loading...'}
                </div>
                <div className={`stat-trend ${modelStatus.initialized ? 'positive' : 'negative'}`}>
                  {modelStatus.initialized ? 'YOLOv8 Running' : 'Initializing'}
                </div>
              </div>
            </div>
            
            <div className="dashboard-buttons">
              <button onClick={openCamera} className="action-button camera-button">
                <span className="icon">📷</span>
                Open Live Camera
              </button>
              
              <button className="action-button reports-button">
                <span className="icon">📊</span>
                Download Reports
              </button>
            </div>
            
            <div className="recent-activity">
              <h3>Recent Detections</h3>
              <table className="detections-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Location</th>
                    <th>Time</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {detectionData.map(detection => (
                    <tr key={detection.id} className={detection.status === 'Spitting Detected' ? 'detection-row-alert' : ''}>
                      <td>{detection.id}</td>
                      <td>{detection.location}</td>
                      <td>{detection.timestamp}</td>
                      <td className={`status-cell ${detection.status === 'Spitting Detected' ? 'status-alert' : 'status-ok'}`}>
                        {detection.status}
                      </td>
                      <td>
                        <button className="view-button">View</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        {activeTab === 'detections' && (
          <div className="detections-tab">
            <h2>Recorded Detections</h2>
            <div className="detections-filters">
              <select defaultValue="all">
                <option value="all">All Locations</option>
                <option value="camera1">Camera 1</option>
                <option value="camera2">Camera 2</option>
                <option value="live">Live Camera</option>
              </select>
              
              <select defaultValue="all">
                <option value="all">All Statuses</option>
                <option value="detected">Spitting Detected</option>
                <option value="clean">Clean</option>
              </select>
              
              <button className="filter-button">Filter</button>
            </div>
            
            <div className="detections-list">
              {detections.length > 0 ? (
                detections.map(detection => (
                  <div className="detection-item" key={detection.id}>
                    <div className="detection-info">
                      <h4>Detection #{detection.id}</h4>
                      <p>Location: {detection.location}</p>
                      <p>Time: {detection.timestamp}</p>
                      <p className={`detection-status ${detection.status === 'Spitting Detected' ? 'status-alert' : 'status-ok'}`}>
                        Status: {detection.status}
                      </p>
                      <p>Severity: {detection.severity}</p>
                      <div className="detection-actions">
                        <button>View Details</button>
                        <button>Download</button>
                      </div>
                    </div>
                    
                    {detection.image && (
                      <div className="detection-image-container">
                        <img src={detection.image} alt={`Detection ${detection.id}`} />
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="no-detections">
                  <p>No detections recorded yet. Start the camera to begin monitoring.</p>
                  <button onClick={openCamera} className="open-camera-button">
                    Start Camera
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
        
        {activeTab === 'settings' && (
          <div className="settings-tab">
            <h2>System Settings</h2>
            
            <div className="settings-group">
              <h3>Camera Settings</h3>
              <div className="setting-item">
                <label>Detection Sensitivity</label>
                <select defaultValue="medium">
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              
              <div className="setting-item">
                <label>Notification Sound</label>
                <select defaultValue="on">
                  <option value="on">On</option>
                  <option value="off">Off</option>
                </select>
              </div>
            </div>
            
            <div className="settings-group">
              <h3>AI Model Settings</h3>
              <div className="setting-item">
                <label>Model Status</label>
                <div className={`model-status ${modelStatus.initialized ? 'model-active' : 'model-inactive'}`}>
                  {modelStatus.initialized ? 'Active' : 'Inactive'}
                </div>
              </div>
              
              <div className="setting-item">
                <label>Model Path</label>
                <input type="text" readOnly value={modelStatus.path || 'Not loaded'} />
              </div>
              
              <div className="setting-item">
                <button 
                  onClick={checkModelStatus} 
                  disabled={modelStatus.loading}
                  className="refresh-button"
                >
                  {modelStatus.loading ? 'Loading...' : 'Refresh Model Status'}
                </button>
              </div>
            </div>
            
            <div className="settings-actions">
              <button className="save-settings">Save Settings</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )

  const renderPage = () => {
    switch (currentPage) {
      case 'login':
        return <LoginPage />
      case 'dashboard':
        return <Dashboard />
      case 'camera':
        return <CameraPage />
      default:
        return <LoginPage />
    }
  }

  return (
    <div className="app-container">
      {renderPage()}
    </div>
  )
}

export default App
