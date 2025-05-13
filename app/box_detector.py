from os import putenv

import numpy as np
import cv2
from .models import load_models, get_emotion_model_details
from .config import EMOTION_LABELS
import insightface
from insightface.app import FaceAnalysis
import os

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

        # Khởi tạo face analyzer từ InsightFace
        self.face_analyzer = None
        try:
            # First try with CUDA if available
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available
            
            # Chuẩn bị thư mục models nếu cần
            models_dir = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_l')
            os.makedirs(models_dir, exist_ok=True)
            
            try:
                # Try with CUDA first
                self.face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                print("Successfully initialized InsightFace with CUDA")
            except Exception as e:
                print(f"Failed to initialize InsightFace with CUDA: {e}")
                # Fall back to CPU
                self.face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
                self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                print("Successfully initialized InsightFace with CPU")
                
        except Exception as e:
            print(f"Failed to initialize InsightFace: {e}")
            # Check if models directory exists
            models_dir = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_l')
            if not os.path.exists(models_dir):
                print(f"InsightFace models not found. Expected at: {models_dir}")
                print("Please download models manually or run: python -m insightface.model_zoo --download-dir=~/.insightface/models/ --download buffalo_l")
        
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
                                    #Nhận diện cảm xúc / Emotion detection
                                    emotion = self._detect_emotion(face_roi)
                                
                                    # Trích xuất vector embedding từ khuôn mặt
                                    face_roi_embedding = self.extract_face_with_margin(original_frame, (px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2), margin_percent=0.2)
                                    embedding = self.get_embedding_from_image(face_roi_embedding)
                                    if embedding is None:
                                        print(f"Could not generate embedding for face")
                                else:
                                    emotion = "Không xác định"  # Không thể xác định cảm xúc / Unknown emotion
                                    embedding = None
                            else:
                                emotion = "Không xác định"
                                embedding = None
                            if embedding is not None:
                                embedding = embedding.tolist()  # Chuyển đổi sang danh sách để dễ dàng lưu trữ
                            
                            # Tọa độ toàn cục cho khuôn mặt / Global coordinates for the face
                            global_coords = (px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2)
                            face_boxes.append((global_coords, conf, emotion, embedding))
                
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

    # Hàm trích xuất khuôn mặt với margin
    def extract_face_with_margin(self, img, bbox, margin_percent=0.2, min_size=112):
        """
        Trích xuất khuôn mặt với margin và đảm bảo kích thước tối thiểu.
        Args:
            img (np.ndarray): Ảnh gốc.
            bbox (list): Bounding box [x1, y1, x2, y2].
            margin_percent (float): Phần trăm margin so với kích thước bbox.
            min_size (int): Kích thước tối thiểu của ảnh khuôn mặt.
        Returns:
            np.ndarray: Ảnh khuôn mặt đã cắt và tiền xử lý.
        """
        height, width = img.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        margin_x = int(w * margin_percent)
        margin_y = int(h * margin_percent)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        face_img = img[y1:y2, x1:x2]
        min_dim = min(face_img.shape[0], face_img.shape[1])
        if min_dim < min_size:
            scale = min_size / min_dim
            new_width = int(face_img.shape[1] * scale)
            new_height = int(face_img.shape[0] * scale)
            face_img = cv2.resize(face_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return face_img

    # Hàm tạo embedding từ ảnh
    def get_embedding_from_image(self, img):
        """
        Tạo embedding từ ảnh đã đọc.
        Args:
            face_analyzer: Đối tượng phân tích khuôn mặt InsightFace.
            img (np.ndarray): Ảnh dạng numpy array (BGR).
        Returns:
            np.ndarray: Vector embedding hoặc None nếu không phát hiện khuôn mặt.
        """
        try:
            if self.face_analyzer is None:
                print("Face analyzer not initialized properly")
                return None
                
            if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print("Invalid image input for embedding generation")
                return None
                
            # Đảm bảo ảnh đủ lớn để phát hiện khuôn mặt
            min_dim = min(img.shape[0], img.shape[1])
            if min_dim < 80:  # InsightFace thường cần ảnh đủ lớn để phát hiện khuôn mặt
                scale = 80 / min_dim
                new_width = int(img.shape[1] * scale)
                new_height = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Thêm debug
            print(f"Processing image with shape {img.shape} for embedding")
            
            faces = self.face_analyzer.get(img)
            if len(faces) == 0:
                print("No faces detected by face_analyzer in the provided image")
                return None
                
            best_face = max(faces, key=lambda x: x.det_score)
            return best_face.embedding
        except Exception as e:
            print(f"Error in get_embedding_from_image: {e}")
            return None

