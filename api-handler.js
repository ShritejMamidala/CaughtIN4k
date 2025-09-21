document.addEventListener('DOMContentLoaded', function() {
    // Initialize API handler
    const apiHandler = new ApiHandler();
    apiHandler.init();
   
    // UI Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const directionsButton = document.getElementById('directions-button');
    const mainContent = document.getElementById('main-content');
    const directionsPage = document.getElementById('directions-page');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const uploadPercentage = document.getElementById('uploadPercentage');
    const uploadStatus = document.getElementById('uploadStatus');
    const analysisProgress = document.getElementById('analysisProgress');
    const analysisProgressBar = document.getElementById('analysisProgressBar');
    const analysisPercentage = document.getElementById('analysisPercentage');
    const analysisStatus = document.getElementById('analysisStatus');
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
   
    // Video Playback Controls Elements
    const videoOverlay = document.getElementById('videoOverlay');
    const previewVideo = document.getElementById('previewVideo');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const rewindBtn = document.getElementById('rewindBtn');
    const forwardBtn = document.getElementById('forwardBtn');
    const slowDownBtn = document.getElementById('slowDownBtn');
    const speedUpBtn = document.getElementById('speedUpBtn');
    const currentTimeEl = document.getElementById('currentTime');
    const totalTimeEl = document.getElementById('totalTime');
    const changeVideoBtn = document.getElementById('changeVideoBtn');
   
    // Function to format time as MM:SS
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
   
    // Set up event listeners for the video
    previewVideo.addEventListener('loadedmetadata', function() {
        totalTimeEl.textContent = formatTime(previewVideo.duration);
    });
   
    previewVideo.addEventListener('timeupdate', function() {
        currentTimeEl.textContent = formatTime(previewVideo.currentTime);
    });
   
    previewVideo.addEventListener('play', function() {
        playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
    });
   
    previewVideo.addEventListener('pause', function() {
        playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
    });
   
    // Event listeners for playback controls
    playPauseBtn.addEventListener('click', function() {
        if (previewVideo.paused) {
            previewVideo.play();
        } else {
            previewVideo.pause();
        }
    });
   
    rewindBtn.addEventListener('click', function() {
        previewVideo.currentTime = Math.max(0, previewVideo.currentTime - 10);
    });
   
    forwardBtn.addEventListener('click', function() {
        previewVideo.currentTime = Math.min(previewVideo.duration, previewVideo.currentTime + 10);
    });
   
    slowDownBtn.addEventListener('click', function() {
        if (previewVideo.playbackRate > 0.25) {
            previewVideo.playbackRate -= 0.25;
        }
    });
   
    speedUpBtn.addEventListener('click', function() {
        if (previewVideo.playbackRate < 2) {
            previewVideo.playbackRate += 0.25;
        }
    });
   
    // Change video button
    changeVideoBtn.addEventListener('click', function() {
        // Restore the blue dotted border
        uploadArea.style.border = '3px dashed #64ffda';
        videoOverlay.classList.add('hidden');
        fileInput.value = '';
        analyzeBtn.disabled = true;
    });
   
    // Navigation between pages with transitions
    directionsButton.addEventListener('click', function(e) {
        e.preventDefault();
       
        // Add a slight bounce effect to the button
        directionsButton.style.transform = 'translateY(2px)';
        setTimeout(() => {
            directionsButton.style.transform = 'translateY(0)';
        }, 150);
       
        if (mainContent.classList.contains('visible')) {
            // Going to directions page - slide current page up, new page up from bottom
            mainContent.classList.remove('visible');
            mainContent.classList.add('hidden');
           
            // Show directions page after a short delay
            setTimeout(() => {
                directionsPage.classList.remove('hidden');
                setTimeout(() => {
                    directionsPage.classList.add('visible');
                    directionsButton.textContent = 'Back to Main Menu';
                }, 50);
            }, 300);
        } else {
            // Going back to main page - slide current page down, new page down from top
            directionsPage.classList.remove('visible');
            directionsPage.classList.add('hidden');
           
            // Show main page after a short delay
            setTimeout(() => {
                mainContent.classList.remove('hidden');
                setTimeout(() => {
                    mainContent.classList.add('visible');
                    directionsButton.textContent = 'Directions';
                }, 50);
            }, 300);
        }
    });
   
    // Open file dialog when clicking the upload button
    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });
   
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const file = this.files[0];
           
            // Check if file is a video
            if (file.type.startsWith('video/')) {
                // Remove the blue dotted border
                uploadArea.style.border = 'none';
               
                // Enable analyze button
                analyzeBtn.disabled = false;
               
                // Set the video source
                previewVideo.src = URL.createObjectURL(file);
               
                // Show the video overlay and hide upload area
                videoOverlay.classList.remove('hidden');
               
                // Reset playback rate
                previewVideo.playbackRate = 1;
            } else {
                alert('Please select a video file.');
            }
        }
    });
   
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.borderColor = '#0a192f';
        this.style.backgroundColor = 'rgba(100, 255, 218, 0.2)';
    });
   
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.borderColor = '#64ffda';
        this.style.backgroundColor = 'rgba(10, 25, 47, 0.3)';
    });
   
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
       
        this.style.borderColor = '#64ffda';
        this.style.backgroundColor = 'rgba(10, 25, 47, 0.3)';
       
        const file = e.dataTransfer.files[0];
       
        // Check if file is a video
        if (file.type.startsWith('video/')) {
            // Remove the blue dotted border
            uploadArea.style.border = 'none';
           
            // Set the file to input
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
           
            // Enable analyze button
            analyzeBtn.disabled = false;
           
            // Set the video source
            previewVideo.src = URL.createObjectURL(file);
           
            // Show the video overlay and hide upload area
            videoOverlay.classList.remove('hidden');
           
            // Reset playback rate
            previewVideo.playbackRate = 1;
        } else {
            alert('Please drop a video file.');
        }
    });
   
    // Initialize page states on load
    window.addEventListener('load', function() {
        mainContent.classList.add('visible');
        directionsPage.classList.add('hidden');
    });
});

// API Handler for Caught in 4K Application
class ApiHandler {
    constructor() {
        this.baseUrl = 'http://localhost:8000'; // FastAPI server URL
        this.isAnalyzing = false;
    }

    // Initialize API handler
    init() {
        console.log('API Handler initialized');
        this.bindAnalyzeButton();
    }

    // Bind the analyze button to use the API
    bindAnalyzeButton() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            // Remove existing click event listeners
            const newAnalyzeBtn = analyzeBtn.cloneNode(true);
            analyzeBtn.parentNode.replaceChild(newAnalyzeBtn, analyzeBtn);
           
            // Add new event listener that uses the API
            newAnalyzeBtn.addEventListener('click', () => {
                this.handleAnalysis();
            });
        }
    }

    // Handle the analysis process with API calls
    async handleAnalysis() {
        const fileInput = document.getElementById('fileInput');
        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please select a video file first.');
            return;
        }

        const file = fileInput.files[0];
        this.isAnalyzing = true;
       
        try {
            // Show upload progress
            this.showUploadProgress();
           
            // Upload the file to the FastAPI server
            const uploadResponse = await this.uploadVideo(file);
           
            if (uploadResponse.success) {
                // Start analysis
                const analysisResponse = await this.startAnalysis(uploadResponse.video_id);
               
                if (analysisResponse.success) {
                    // Poll for results
                    await this.pollAnalysisResults(analysisResponse.analysis_id);
                } else {
                    throw new Error('Failed to start analysis');
                }
            } else {
                throw new Error('Failed to upload video');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            alert('Analysis failed: ' + error.message);
            this.resetUI();
        } finally {
            this.isAnalyzing = false;
        }
    }

    // Show upload progress UI
    showUploadProgress() {
        const uploadProgress = document.getElementById('uploadProgress');
        const analysisProgress = document.getElementById('analysisProgress');
        const resultsPlaceholder = document.getElementById('resultsPlaceholder');
       
        if (uploadProgress) uploadProgress.style.display = 'block';
        if (analysisProgress) analysisProgress.style.display = 'none';
        if (resultsPlaceholder) resultsPlaceholder.style.display = 'none';
       
        // Initialize progress bar
        const uploadProgressBar = document.getElementById('uploadProgressBar');
        const uploadPercentage = document.getElementById('uploadPercentage');
        const uploadStatus = document.getElementById('uploadStatus');
       
        if (uploadProgressBar) uploadProgressBar.style.width = '0%';
        if (uploadPercentage) uploadPercentage.textContent = '0%';
        if (uploadStatus) uploadStatus.textContent = 'Uploading video...';
    }

    // Upload video to FastAPI server
    async uploadVideo(file) {
        const formData = new FormData();
        formData.append('file', file);
       
        try {
            const response = await fetch(`${this.baseUrl}/upload`, {
                method: 'POST',
                body: formData,
            });
           
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
           
            // Update UI to show upload complete
            const uploadProgressBar = document.getElementById('uploadProgressBar');
            const uploadPercentage = document.getElementById('uploadPercentage');
            const uploadStatus = document.getElementById('uploadStatus');
           
            if (uploadProgressBar) uploadProgressBar.style.width = '100%';
            if (uploadPercentage) uploadPercentage.textContent = '100%';
            if (uploadStatus) uploadStatus.textContent = 'Upload complete!';
           
            return await response.json();
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    }

    // Start analysis on the uploaded video
    async startAnalysis(videoId) {
        try {
            const response = await fetch(`${this.baseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ video_id: videoId }),
            });
           
            if (!response.ok) {
                throw new Error(`Analysis start failed: ${response.statusText}`);
            }
           
            return await response.json();
        } catch (error) {
            console.error('Analysis start error:', error);
            throw error;
        }
    }

    // Poll for analysis results
    async pollAnalysisResults(analysisId) {
        let attempts = 0;
        const maxAttempts = 100; // Prevent infinite polling
       
        while (attempts < maxAttempts && this.isAnalyzing) {
            try {
                const response = await fetch(`${this.baseUrl}/results/${analysisId}`);
               
                if (!response.ok) {
                    throw new Error(`Results fetch failed: ${response.statusText}`);
                }
               
                const data = await response.json();
               
                // Update progress UI
                this.updateAnalysisProgress(data.progress || 0, data.status || 'Processing');
               
                if (data.status === 'completed') {
                    // Display results
                    this.displayResults(data.results);
                    return;
                } else if (data.status === 'failed') {
                    throw new Error('Analysis failed on server');
                }
               
                // Wait before polling again
                await new Promise(resolve => setTimeout(resolve, 2000));
                attempts++;
            } catch (error) {
                console.error('Polling error:', error);
                throw error;
            }
        }
       
        if (attempts >= maxAttempts) {
            throw new Error('Analysis timed out');
        }
    }

    // Update analysis progress UI
    updateAnalysisProgress(progress, status) {
        const analysisProgress = document.getElementById('analysisProgress');
        const analysisProgressBar = document.getElementById('analysisProgressBar');
        const analysisPercentage = document.getElementById('analysisPercentage');
        const analysisStatus = document.getElementById('analysisStatus');
        const uploadProgress = document.getElementById('uploadProgress');
       
        if (uploadProgress) uploadProgress.style.display = 'none';
        if (analysisProgress) analysisProgress.style.display = 'block';
        if (analysisProgressBar) analysisProgressBar.style.width = `${progress}%`;
        if (analysisPercentage) analysisPercentage.textContent = `${progress}%`;
        if (analysisStatus) analysisStatus.textContent = status;
    }

    // Display analysis results
    displayResults(results) {
        const analysisProgress = document.getElementById('analysisProgress');
        const resultsPlaceholder = document.getElementById('resultsPlaceholder');
       
        if (analysisProgress) analysisProgress.style.display = 'none';
        if (resultsPlaceholder) {
            resultsPlaceholder.style.display = 'block';
            resultsPlaceholder.innerHTML = `
                <div class="results-icon">
                    <i class="fas fa-check-circle"></i>
                </div>
                <h3>Analysis Complete</h3>
                <div class="results-content">
                    ${this.formatResults(results)}
                </div>
            `;
        }
    }

    // Format results for display
// Format results for display
formatResults(results) {
  if (!results || results.length === 0) {
    return '<p>No criminal activities detected.</p>';
  }

  const fmt = (x, d=1) => Number.isFinite(+x) ? (+x).toFixed(d) : '—';

  let html = '<div class="detected-activities">';
  html += '<h4>Detected Activities:</h4>';
  html += '<ul class="activities-list">';

  results.forEach(activity => {
    // Time: prefer formatted strings from API; fallback to seconds if needed
    const startStr = activity.start_time
      || (Number.isFinite(activity.start_seconds) ? this.formatTime(activity.start_seconds) : '00:00');
    const endStr = activity.end_time
      || (Number.isFinite(activity.end_seconds) ? this.formatTime(activity.end_seconds) : '00:00');

    // Confidence: prefer percentage from API; else compute from raw (0–1) or old confidence field
    let confPct;
    if (typeof activity.confidence_pct === 'number') {
      confPct = activity.confidence_pct;
    } else if (typeof activity.confidence_raw === 'number') {
      confPct = activity.confidence_raw * 100;
    } else if (typeof activity.confidence === 'number') {
      // old schema (0–1). If your backend already sent %, remove the *100 here.
      confPct = activity.confidence * 100;
    } else {
      confPct = 0;
    }

    html += `
      <li class="activity-item">
        <div class="activity-type">${activity.type || 'Unknown'}</div>
        <div class="activity-time">Timestamp: ${startStr} – ${endStr}</div>
        <div class="activity-confidence">Confidence: ${fmt(confPct, 1)}%</div>
      </li>
    `;
  });

  html += '</ul></div>';
  return html;
}


    // Format time as MM:SS
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    // Reset UI to initial state
    resetUI() {
        const uploadProgress = document.getElementById('uploadProgress');
        const analysisProgress = document.getElementById('analysisProgress');
        const resultsPlaceholder = document.getElementById('resultsPlaceholder');
       
        if (uploadProgress) uploadProgress.style.display = 'none';
        if (analysisProgress) analysisProgress.style.display = 'none';
        if (resultsPlaceholder) {
            resultsPlaceholder.style.display = 'block';
            resultsPlaceholder.innerHTML = `
                <div class="results-icon">
                    <i class="fas fa-hourglass-half"></i>
                </div>
                <p class="results-text">Analysis results will appear here after processing</p>
            `;
        }
    }
}