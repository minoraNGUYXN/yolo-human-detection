/**
 * ui.js
 * Handle UI interactions and event listeners
 */

import state from './state.js';
import config from './config.js';
import { startCamera, stopCamera } from './camera.js';
import { drawDetections } from './detection.js';
import { resetStats } from './stats.js';

// DOM elements
let video, overlay, ctx, cameraSelect, toggleRearCamera;
let togglePerson, toggleFace, toggleConfidence, toggleEmotions;

/**
 * Initialize UI module
 * @param {Object} elements - DOM elements
 */
export function initUI(elements) {
    video = elements.video;
    overlay = elements.overlay;
    ctx = overlay.getContext('2d');
    cameraSelect = elements.cameraSelect;
    toggleRearCamera = elements.toggleRearCamera;
    togglePerson = elements.togglePerson;
    toggleFace = elements.toggleFace;
    toggleConfidence = elements.toggleConfidence;
    toggleEmotions = elements.toggleEmotions;
}

/**
 * Setup all event listeners
 */
export function setupEventListeners() {
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    
    startButton.addEventListener('click', startCamera);
    stopButton.addEventListener('click', stopCamera);
    
    // Camera toggle event listener
    toggleRearCamera.addEventListener('change', () => {
        // Disable camera select dropdown when rear camera is toggled
        cameraSelect.disabled = toggleRearCamera.checked;
        
        // Restart camera with new configuration if running
        if (state.isRunning) {
            stopCamera();
            startCamera();
        }
    });
    
    // Camera select dropdown event listener
    cameraSelect.addEventListener('change', () => {
        // Restart camera with new configuration if running
        if (state.isRunning) {
            stopCamera();
            startCamera();
        }
    });
    
    // Detection display toggles
    togglePerson.addEventListener('change', () => {
        config.showPersons = togglePerson.checked;
        if (state.isRunning) {
            // Redraw immediately with new settings
            drawDetections(state.personBoxes, state.faceBoxes);
        }
    });
    
    toggleFace.addEventListener('change', () => {
        config.showFaces = toggleFace.checked;
        if (state.isRunning) {
            // Redraw immediately with new settings
            drawDetections(state.personBoxes, state.faceBoxes);
        }
    });
    
    toggleConfidence.addEventListener('change', () => {
        config.showConfidence = toggleConfidence.checked;
        if (state.isRunning) {
            // Redraw immediately with new settings
            drawDetections(state.personBoxes, state.faceBoxes);
        }
    });
    
    // Thêm xử lý sự kiện cho hiển thị cảm xúc / Add emotion toggle event listener
    toggleEmotions.addEventListener('change', () => {
        config.showEmotions = toggleEmotions.checked;
        if (state.isRunning) {
            // Redraw immediately with new settings
            drawDetections(state.personBoxes, state.faceBoxes);
        }
    });

    // Thêm xử lý sự kiện cho hiển thị danh tính / Add identity toggle event listener
    toggleIdentity.addEventListener('change', () => {
    config.showIdentity = toggleIdentity.checked;
    if (state.isRunning) {
        drawDetections(state.personBoxes, state.faceBoxes);
    }
});
}

/**
 * Configure UI based on device type (mobile/desktop)
 */
export function configureUIForDevice() {
    const cameraSelectContainer = document.getElementById('cameraSelectContainer');
    const toggleCameraContainer = document.getElementById('toggleCameraContainer');
    
    if (config.isMobile) {
        // On mobile: Hide dropdown, show toggle
        cameraSelectContainer.style.display = 'none';
        toggleCameraContainer.style.display = 'block';
    } else {
        // On desktop: Show dropdown, hide toggle
        cameraSelectContainer.style.display = 'block';
        toggleCameraContainer.style.display = 'none';
    }
}

/**
 * Reset canvas to initial state
 */
export function resetCanvas() {
    // Ensure canvas has correct dimensions
    const container = document.getElementById('videoContainer');
    overlay.width = container.clientWidth;
    overlay.height = container.clientHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    
    // Reset statistics
    resetStats();
    
    // Reset detection boxes
    state.personBoxes = [];
    state.faceBoxes = [];
}