/**
 * detection.js
 * Handle frame processing and drawing detection results
 */

import state from './state.js';
import config from './config.js';
import { updateStats, setupFpsCounter } from './stats.js';

// DOM elements
let video, overlay, ctx;

/**
 * Initialize detection module
 * @param {Object} elements - DOM elements
 */
export function initDetection(elements) {
    video = elements.video;
    overlay = elements.overlay;
    ctx = overlay.getContext('2d');
}

/**
 * Start frame processing
 */
export function startProcessing() {
    if (state.isProcessing) return;
    
    state.isProcessing = true;
    
    // Setup FPS counter
    setupFpsCounter();
    
    // Start processing frames
    processFrame();
}

/**
 * Stop frame processing
 */
export function stopProcessing() {
    state.isProcessing = false;
    
    // Cancel pending processing timer
    if (state.processingTimerId) {
        clearTimeout(state.processingTimerId);
        state.processingTimerId = null;
    }
}

/**
 * Process a single frame and schedule the next one
 */
export async function processFrame() {
    // Check if processing is still active
    if (!state.isProcessing || !state.isRunning) {
        return;
    }
    
    const startTime = performance.now();
    
    try {
        // Capture frame from video
        const canvas = document.createElement('canvas');
        const tempCtx = canvas.getContext('2d');
        
        // Check if video is ready
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            // Video not ready, retry later
            state.processingTimerId = setTimeout(processFrame, 100);
            return;
        }
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current frame to temporary canvas
        tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });
        
        // Prepare data to send
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        // Prepare URL with query parameters
        let apiUrl = new URL(config.serverUrl);
        
        // Add query parameters for identification if enabled
        if (config.enableIdentification) {
            apiUrl.searchParams.append('identify', 'true');
            apiUrl.searchParams.append('threshold', config.identificationThreshold.toString());
            apiUrl.searchParams.append('top_k', config.maxMatchResults.toString());
        }

        // Send frame to server for processing
        const response = await fetch(config.serverUrl, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Process server response
        const result = await response.json();
        
        // Save results to application state
        state.personBoxes = result.person_boxes || [];
        state.faceBoxes = result.face_boxes || [];
        state.identifiedCount = result.identified_count || 0;

        // Draw detection results
        drawDetections(state.personBoxes, state.faceBoxes);
        
        // Update statistics
        updateStats(result.persons, result.faces, state.faceBoxes, state.identifiedCount);
        
        // Increment frame counter for FPS calculation
        state.frameCount++;
        
        // Calculate frame processing time
        const processingTime = performance.now() - startTime;
        
        // Calculate delay for desired frame rate
        const targetFrameTime = 1000 / config.frameRate;
        const delayTime = Math.max(0, targetFrameTime - processingTime);
        
        // Schedule next frame if application is still running
        if (state.isProcessing && state.isRunning) {
            state.processingTimerId = setTimeout(processFrame, delayTime);
        }
        
    } catch (error) {
        console.error('Lỗi khi xử lý frame:', error);
        
        // If error occurs, retry after 1 second if application is still running
        if (state.isProcessing && state.isRunning) {
            state.processingTimerId = setTimeout(processFrame, 1000);
        }
    }
}

/**
 * Draw detection boxes for persons and faces
 * @param {Array} personBoxes - Person detection boxes
 * @param {Array} faceBoxes - Face detection boxes with emotion data
 */
export function drawDetections(personBoxes, faceBoxes) {
    // Only draw when application is running
    if (!state.isRunning) {
        return;
    }
    
    // Clear canvas before drawing
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    
    // Get font size from config
    const fontSize = config.isMobile ? 
                    (config.mobileLabelFontSize || 14) : 
                    (config.desktopLabelFontSize || 16);
    
    // Calculate label height and padding
    const labelPadding = config.labelPadding || 6;
    const labelMargin = config.labelMargin || 8;
    const labelHeight = fontSize + labelPadding * 2;
    const borderWidth = config.borderWidth || 4; // Độ dày của đường viền
    
    // Draw person boxes if enabled
    if (config.showPersons && personBoxes && personBoxes.length > 0) {
        personBoxes.forEach(box => {
            const [x1, y1, x2, y2] = box.coords;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Draw person box
            ctx.strokeStyle = config.personColor;
            ctx.lineWidth = borderWidth;
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label if confidence display is enabled
            if (config.showConfidence) {
                // Set font size for label
                ctx.font = `bold ${fontSize}px Arial`;
                
                // Create label text
                const label = `Người ${box.confidence.toFixed(2)}`;
                const textWidth = ctx.measureText(label).width + labelPadding * 2;
                
                // Create label background
                ctx.fillStyle = config.personColor;
                ctx.fillRect(x1, y1 - labelHeight - labelMargin, textWidth, labelHeight);
                
                // Draw text
                ctx.fillStyle = '#FFFFFF';
                ctx.fillText(label, x1 + labelPadding, y1 - labelMargin - labelPadding);
            }
        });
    }
    
    // Draw face boxes if enabled
    if (config.showFaces && faceBoxes && faceBoxes.length > 0) {
        faceBoxes.forEach(box => {
            const [x1, y1, x2, y2] = box.coords;
            const width = x2 - x1;
            const height = y2 - y1;

            // Determine color based on identification status
            let boxColor = config.faceColor;
            if (config.enableIdentification && box.match && box.match.length > 0) {
                boxColor = config.identifiedFaceColor || '#00FF00'; // Default to green if not specified
            }
            
            // Draw face box
            ctx.strokeStyle = config.faceColor;
            ctx.lineWidth = borderWidth;
            ctx.strokeRect(x1, y1, width, height);
            
            // Compose label text
            let labelParts = [];
            if (config.showConfidence) {
                labelParts.push(`Mặt ${box.confidence.toFixed(2)}`);
            }
            
            // Add emotion if enabled and available
            if (config.showEmotions && box.emotion) {
                labelParts.push(box.emotion);
                
                // Use emotion-specific color if available
                if (config.emotionColors[box.emotion]) {
                    ctx.fillStyle = config.emotionColors[box.emotion];
                } else {
                    ctx.fillStyle = config.emotionColor;
                }
            } else {
                ctx.fillStyle = config.faceColor;
            }
            
            // Add identification info if available
            if (config.enableIdentification && config.showIdentity && box.match && box.match.length > 0) {
                // Use the top match (highest similarity)
                const topMatch = box.match[0];
                const name = topMatch.metadata.name || 'Unknown';
                const similarity = (topMatch.distance * 100).toFixed(0);
                
                labelParts.push(`${name} (${similarity}%)`);
                
                // Use identified color for label background
                ctx.fillStyle = config.identifiedFaceColor || '#00FF00';
            }

            // Only draw label if we have something to show
            if (labelParts.length > 0) {
                // Set font size for label
                ctx.font = `bold ${fontSize}px Arial`;
                
                const label = labelParts.join(' - ');
                const textWidth = ctx.measureText(label).width + labelPadding * 2;
                
                // Create label background (position above the face box)
                ctx.fillRect(x1, y1 - labelHeight - labelMargin, textWidth, labelHeight);
                
                // Draw text
                ctx.fillStyle = '#FFFFFF';
                ctx.fillText(label, x1 + labelPadding, y1 - labelMargin - labelPadding);
            }
        });
    }
}