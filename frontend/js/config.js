/**
 * config.js
 * Configuration settings for the application
 */

// Check if the device is mobile
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Export application settings
const config = {
    // Detection settings
    showPersons: true,
    showFaces: true,
    showConfidence: true,
    showEmotions: true,  // Hiển thị cảm xúc / Show emotions
    showIdentity: true,
    personColor: '#e74c3c',
    faceColor: '#2ecc71',
    emotionColor: '#ff9800',  // Màu mặc định cho cảm xúc / Default color for emotions
    
    // Màu sắc cho từng cảm xúc / Colors for different emotions
    emotionColors: {
        'Giận dữ': '#e74c3c',     // Đỏ / Red
        'Ghê tởm': '#9b59b6',     // Tím / Purple
        'Sợ hãi': '#34495e',      // Xám đậm / Dark gray
        'Vui vẻ': '#f1c40f',      // Vàng / Yellow
        'Buồn bã': '#3498db',     // Xanh dương / Blue
        'Ngạc nhiên': '#e67e22',  // Cam / Orange
        'Bình thường': '#95a5a6', // Xám nhạt / Light gray
        'Không xác định': '#7f8c8d' // Xám / Gray
    },
    
    // Server settings
    serverUrl: 'http://localhost:8000/process_frame',
    
    // Performance settings
    frameRate: isMobile ? 10 : 30, // Lower FPS on mobile devices
    
    // Device info
    isMobile: isMobile,
    
    // Nhãn kết quả nhận diện / Detection label settings
    desktopLabelFontSize: 16,     // Kích thước chữ trên desktop (px)
    mobileLabelFontSize: 14,      // Kích thước chữ trên thiết bị di động (px)
    labelPadding: 6,              // Padding cho nhãn (px)
    labelMargin: 8,               // Khoảng cách từ nhãn đến khung (px)
    borderWidth: 4                // Độ dày viền khung (px)
};

export default config;