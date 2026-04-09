import { useEffect, useRef, useState } from 'react'

function App() {
  const [mode, setMode] = useState('live') // 'live' or 'upload'
  
  // LIVE State
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [backendStatus, setBackendStatus] = useState('Idle')

  // UPLOAD State
  const fileInputRef = useRef(null)
  const photoCanvasRef = useRef(null)
  const imgRef = useRef(null)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [isScanning, setIsScanning] = useState(false)

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    setBackendStatus('Idle')
  }

  const startCamera = async () => {
    // Ask for permission every single time
    const userConsent = window.confirm("EmotionTrack is requesting to turn on your webcam. Do you want to proceed?")
    if (!userConsent) return

    try {
      setUploadedImage(null) // clear upload
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if(videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      setIsStreaming(true)
      setBackendStatus('Connecting...')
    } catch (err) {
      alert("Could not access camera.")
    }
  }

  // Handle Mode Switch
  const switchMode = (m) => {
    setMode(m)
    if (m === 'upload') {
      stopCamera()
    } else {
      setUploadedImage(null)
    }
  }

  // --- LIVE CAMERA LOOP ---
  useEffect(() => {
    let intervalId;

    if (isStreaming && mode === 'live') {
      intervalId = setInterval(async () => {
        const video = videoRef.current
        const canvas = canvasRef.current
        if (!video || !canvas) return

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }

        const ctx = canvas.getContext('2d')
        const offCanvas = document.createElement('canvas')
        offCanvas.width = video.videoWidth
        offCanvas.height = video.videoHeight
        const offCtx = offCanvas.getContext('2d')
        
        offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height)
        const dataUrl = offCanvas.toDataURL('image/jpeg', 0.8)

        try {
          const res = await fetch('http://127.0.0.1:5000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
          })
          if (!res.ok) throw new Error("API error")
          
          const data = await res.json()
          setBackendStatus('Tracking Active')

          ctx.clearRect(0, 0, canvas.width, canvas.height)

          if (data.faces) {
            data.faces.forEach(face => {
              const [x, y, w, h] = face.box
              ctx.strokeStyle = '#38bdf8'
              ctx.lineWidth = 4
              ctx.strokeRect(x, y, w, h)
              ctx.fillStyle = '#38bdf8'
              ctx.fillRect(x, y - 40, w, 40)

              // 🌟 FIX FOR INVERTED TEXT
              ctx.save()
              ctx.scale(-1, 1) // Flip context horizontally
              ctx.fillStyle = '#ffffff'
              ctx.font = 'bold 24px Inter'
              // Draw text backward so the CSS transform un-mirrors it properly!
              ctx.fillText(`${face.emotion} ${Math.round(face.confidence*100)}%`, -x - w + 10, y - 10)
              ctx.restore()
            })
          }
        } catch (err) {
          console.error(err)
        }
      }, 500)
    }
    return () => clearInterval(intervalId)
  }, [isStreaming, mode])

  // --- UPLOAD LOGIC ---
  const handleUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = async (ev) => {
      const dataUrl = ev.target.result
      setUploadedImage(dataUrl)
      setIsScanning(true)

      try {
        const res = await fetch('http://127.0.0.1:5000/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl })
        })
        const data = await res.json()
        
        // Wait 1.5 seconds for visual "cool transition" effect
        setTimeout(() => {
          setIsScanning(false)
          drawStaticFaces(data.faces)
        }, 1500)
        
      } catch (err) {
        setIsScanning(false)
      }
    }
    reader.readAsDataURL(file)
  }

  // Draw bounding boxes strictly over the static uploaded photo
  const drawStaticFaces = (faces) => {
    if (!photoCanvasRef.current || !imgRef.current) return
    const canvas = photoCanvasRef.current
    const img = imgRef.current
    const ctx = canvas.getContext('2d')
    
    // Canvas matches CSS rendered width/height exactly
    const rect = img.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    const scaleX = rect.width / img.naturalWidth
    const scaleY = rect.height / img.naturalHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (faces) {
      faces.forEach(face => {
        const [rawX, rawY, rawW, rawH] = face.box
        const x = rawX * scaleX
        const y = rawY * scaleY
        const w = rawW * scaleX
        const h = rawH * scaleY

        ctx.strokeStyle = '#38bdf8'
        ctx.lineWidth = 4
        ctx.shadowColor = '#38bdf8'
        ctx.shadowBlur = 15
        ctx.strokeRect(x, y, w, h)

        ctx.shadowBlur = 0
        ctx.fillStyle = '#38bdf8'
        ctx.fillRect(x, y - 40, w, 40)
        
        ctx.fillStyle = '#ffffff'
        ctx.font = 'bold 24px Inter'
        // No mirroring hack needed here because uploaded photos aren't mirrored!
        ctx.fillText(`${face.emotion} ${Math.round(face.confidence*100)}%`, x + 10, y - 10)
      })
    }
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>EmotionTrack Engine</h1>
        <p>A real-time PyTorch Facial Recognition Interface</p>
      </div>

      <div className="glass-panel">
        
        {/* Mode Switcher */}
        <div className="mode-switcher">
          <button 
            className={`mode-btn ${mode === 'live' ? 'active' : ''}`} 
            onClick={() => switchMode('live')}
          >
            Live Camera
          </button>
          <button 
            className={`mode-btn ${mode === 'upload' ? 'active' : ''}`} 
            onClick={() => switchMode('upload')}
          >
            Photo Scanner
          </button>
        </div>

        {/* Live Camera Interface */}
        {mode === 'live' && (
          <>
            <div className="controls">
              {!isStreaming ? (
                <button className="btn primary" onClick={startCamera}>Initialize Camera</button>
              ) : (
                <button className="btn danger" onClick={stopCamera}>Shut Down Camera</button>
              )}
              <span className={`status-badge ${isStreaming ? 'live' : ''}`}>
                Backend Status: {backendStatus}
              </span>
            </div>

            <div className="video-container" style={{marginTop: '1.5rem', display: isStreaming ? 'block' : 'none'}}>
              <video ref={videoRef} className="video-feed" autoPlay playsInline muted />
              <canvas ref={canvasRef} className="tracking-canvas" />
            </div>
          </>
        )}

        {/* Upload Mode Interface */}
        {mode === 'upload' && (
          <div style={{width: '100%'}}>
            {!uploadedImage ? (
              <div className="upload-zone" onClick={() => fileInputRef.current.click()}>
                <svg width="64" height="64" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{marginBottom: '1rem'}}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                <h3>Click to Upload Photo</h3>
                <p>or drag and drop an image to be scanned by the neural net.</p>
                <input ref={fileInputRef} type="file" className="upload-input" accept="image/*" onChange={handleUpload} />
              </div>
            ) : (
              <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem'}}>
                <div className="photo-preview-container">
                  <img ref={imgRef} src={uploadedImage} alt="Uploaded" className="photo-preview" />
                  <canvas ref={photoCanvasRef} className="static-canvas" />
                  
                  {isScanning && (
                    <>
                      <div className="scanner-overlay">ANALYZING MICRO-EXPRESSIONS...</div>
                      <div className="scanner"></div>
                    </>
                  )}
                </div>
                
                <div className="controls">
                  <button className="btn danger" onClick={() => fileInputRef.current.click()}>
                    Scan Another Image
                  </button>
                  <input ref={fileInputRef} type="file" className="upload-input" accept="image/*" onChange={handleUpload} />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
