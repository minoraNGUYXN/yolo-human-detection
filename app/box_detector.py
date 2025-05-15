from os import putenv

# Cấu hình môi trường cho AMD GPU / Environment setup for AMD GPUs
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.4.0/")

import numpy as np
import cv2
from .models import load_models, get_emotion_model_details
from .config import EMOTION_LABELS

class Detector:
    """Xử lý nhận diện người, khuôn mặt và cảm xúc / Detection handler"""
    def __init__(self):
        """
        Khởi tạo Detector bằng cách tải các mô hình.
        Initializes the Detector by loading the models.
        """
        # Tải các mô hình / Load models
        self.person_model, self.face_model, self.emotion_interpreter = load_models()
        
        # Lấy thông tin chi tiết về mô hình cảm xúc / Get emotion model details
        self.emotion_input_details, self.emotion_output_details = get_emotion_model_details(self.emotion_interpreter)
        
        # Lấy kích thước đầu vào của mô hình cảm xúc / Get emotion model input size
        self.emotion_input_shape = self.emotion_input_details[0]['shape']
        self.emotion_height = self.emotion_input_shape[1]
        self.emotion_width = self.emotion_input_shape[2]
    
    def process_frame(self, frame):
        """
        Xử lý khung hình và trả về kết quả nhận diện.
        Processes a frame and returns the detection results.

        Args:
            frame (np.ndarray): Khung hình đầu vào (ảnh). Input frame (image).

        Returns:
            tuple: Một tuple chứa:
                   - person_count (int): Số lượng người được nhận diện. Number of detected persons.
                   - face_count (int): Số lượng khuôn mặt được nhận diện. Number of detected faces.
                   - person_boxes (list): Danh sách các khung người. List of person bounding boxes.
                   - face_boxes (list): Danh sách các khung khuôn mặt với thông tin cảm xúc. List of face bounding boxes with emotion info.
        """
        try:
            # Nhận diện người / Detect persons
            person_results = self.person_model(frame, classes=[0], conf=0.3, imgsz=640, half=True, verbose=False)
            
            person_count = 0
            face_count = 0
            face_boxes = []  # Danh sách khung khuôn mặt / Face boxes list
            person_boxes = []
            
            if person_results:
                for person in person_results:
                    # Lấy tọa độ khung người / Get person boxes
                    boxes = person.boxes.xyxy.cpu().numpy() #boxes is a numpy array with shape (n,4)
                    confs = person.boxes.conf.cpu().numpy() #confidence for person box
                    person_count += len(boxes)

                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        person_boxes.append(((x1, y1, x2, y2), float(conf)))
                    
                    # Chuẩn bị vùng quan tâm (ROI) / Prepare ROIs
                    valid_rois, valid_indices = self._prepare_rois(frame, boxes)
                    # Nhận diện khuôn mặt và cảm xúc trong ROI / Detect faces and emotions in ROIs
                    face_data = self._detect_faces_and_emotions(frame, valid_rois, valid_indices, boxes)
                    face_count += face_data["count"]
                    face_boxes.extend(face_data["boxes"])
            
            return person_count, face_count, person_boxes, face_boxes
            
        except Exception as e:
            # Xử lý lỗi tại đây để tránh gây đơ ứng dụng / Handle errors to avoid freezing
            print(f"Error in process_frame: {e}")
            # Trả về giá trị mặc định an toàn / Return safe default values
            return 0, 0, [], []
    
    def _prepare_rois(self, frame, boxes):
        """
        Chuẩn bị các vùng ảnh hợp lệ từ khung người.
        Extracts valid image regions from person bounding boxes.
        """
        valid_rois = []
        valid_indices = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if (x2 > x1) and (y2 > y1) and (x1 >= 0) and (y1 >= 0):
                valid_rois.append(frame[y1:y2, x1:x2])  # Cắt vùng ảnh / Crop region
                valid_indices.append(i)
        return valid_rois, valid_indices
    
    def _detect_faces_and_emotions(self, original_frame, rois, indices, boxes):
        """
        Nhận diện khuôn mặt, cảm xúc và tính toán tọa độ toàn cục.
        Detects faces and emotions and calculates global coordinates.

        Args:
            original_frame (np.ndarray): Khung hình gốc. Original frame.
            rois (list): Danh sách các vùng ảnh (ROI). List of image regions (ROIs).
            indices (list): Danh sách các chỉ số tương ứng với các khung người. List of indices corresponding to person boxes.
            boxes (np.ndarray): Mảng các khung người. Array of person bounding boxes.

        Returns:
            dict: Một dictionary chứa:
                  - count (int): Số lượng khuôn mặt được nhận diện. Number of detected faces.
                  - boxes (list): Danh sách các khung khuôn mặt với thông tin cảm xúc. List of face bounding boxes with emotion info.
        """
        face_boxes = []
        face_count = 0
        
        if not rois:
            return {"count": 0, "boxes": []}
        
        try:
            # Thay đổi: Xử lý từng ROI riêng biệt thay vì dùng batch processing
            for idx, roi in enumerate(rois):
                try:
                    # Xử lý từng ROI một
                    face_result = self.face_model(roi, conf=0.3, imgsz=160, half=True, verbose=False)
                    
                    if len(face_result[0].boxes) > 0:
                        face_count += len(face_result[0].boxes)
                        orig_idx = indices[idx]
                        px1, py1 = int(boxes[orig_idx][0]), int(boxes[orig_idx][1])
                        
                        # Duyệt qua từng khuôn mặt phát hiện được / Iterate through each detected face
                        for box in face_result[0].boxes:
                            fx1, fy1, fx2, fy2 = map(int, box.xyxy[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            
                            # Lấy vùng ảnh khuôn mặt để nhận diện cảm xúc / Get face ROI for emotion detection
                            if fx1 < fx2 and fy1 < fy2 and fx1 >= 0 and fy1 >= 0 and fx2 <= roi.shape[1] and fy2 <= roi.shape[0]:
                                face_roi = roi[fy1:fy2, fx1:fx2]
                                
                                # Chỉ xử lý nếu vùng ảnh khuôn mặt hợp lệ / Only process if face ROI is valid
                                if face_roi.size > 0 and face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                                    emotion = self._detect_emotion(face_roi)
                                else:
                                    emotion = "Không xác định"  # Không thể xác định cảm xúc / Unknown emotion
                            else:
                                emotion = "Không xác định"
                            
                            # Tọa độ toàn cục cho khuôn mặt / Global coordinates for the face
                            global_coords = (px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2)
                            face_boxes.append((global_coords, conf, emotion))
                
                except Exception as e:
                    print(f"Error processing ROI {idx}: {e}")
                    continue  # Bỏ qua ROI này và tiếp tục với ROI tiếp theo
        
        except Exception as e:
            print(f"Error in _detect_faces_and_emotions: {e}")
            return {"count": 0, "boxes": []}
            
        return {"count": face_count, "boxes": face_boxes}
    
    def _detect_emotion(self, face_img):
        """
        Phát hiện cảm xúc từ ảnh khuôn mặt sử dụng mô hình TFLite
        Detect emotion from a face image using the TFLite model
        
        Args:
            face_img (np.ndarray): Vùng ảnh khuôn mặt / Face image region
            
        Returns:
            str: Nhãn cảm xúc dự đoán / Predicted emotion label
        """
        try:
            # Tiền xử lý ảnh khuôn mặt cho mô hình cảm xúc / Preprocess face image for emotion model
            
            # Kiểm tra xem cần chuyển sang ảnh xám không / Check if grayscale is needed
            if self.emotion_input_shape[-1] == 1:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Thay đổi kích thước ảnh theo yêu cầu đầu vào / Resize to expected input dimensions
            resized_face = cv2.resize(face_img, (self.emotion_width, self.emotion_height))
            
            # Kiểm tra kiểu dữ liệu đầu vào từ chi tiết / Check input type from details
            input_dtype = self.emotion_input_details[0]['dtype']
            
            # Xử lý tùy thuộc vào kiểu dữ liệu đầu vào / Process based on input data type
            if input_dtype == np.float32:
                # Chuẩn hóa giá trị pixel cho mô hình float / Normalize pixel values for float models
                normalized_face = resized_face.astype(np.float32) / 255.0
            elif input_dtype == np.uint8:
                # Đối với mô hình lượng tử hóa, giữ nguyên dạng uint8 / For quantized models, keep as uint8
                normalized_face = resized_face.astype(np.uint8)
            else:
                # Mặc định chuẩn hóa kiểu float32 / Default to float32 normalization
                normalized_face = resized_face.astype(np.float32) / 255.0
            
            # Thay đổi hình dạng để phù hợp với đầu vào / Reshape to match input tensor shape
            if self.emotion_input_shape[-1] == 1:
                input_tensor = normalized_face.reshape(1, self.emotion_height, self.emotion_width, 1)
            else:
                input_tensor = normalized_face.reshape(1, self.emotion_height, self.emotion_width, 3)
            
            # Đặt tensor đầu vào / Set input tensor
            self.emotion_interpreter.set_tensor(self.emotion_input_details[0]['index'], input_tensor)
            
            # Chạy suy luận / Run inference
            self.emotion_interpreter.invoke()
            
            # Lấy tensor đầu ra / Get output tensor
            output_tensor = self.emotion_interpreter.get_tensor(self.emotion_output_details[0]['index'])
            
            # Lấy cảm xúc dự đoán / Get predicted emotion
            emotion_idx = np.argmax(output_tensor)
            
            # Đảm bảo chỉ số nằm trong giới hạn của danh sách nhãn / Ensure index is within range of labels
            if 0 <= emotion_idx < len(EMOTION_LABELS):
                emotion_label = EMOTION_LABELS[emotion_idx]
            else:
                emotion_label = "Không xác định"
            
            return emotion_label
            
        except Exception as e:
            print(f"Lỗi khi nhận diện cảm xúc: {e}")
            return "Không xác định"