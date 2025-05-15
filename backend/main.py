from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import os
import sys

import numpy as np
import cv2
import base64


# Thêm đường dẫn để có thể import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.chroma_manager import ChromaFaceDB
from app.box_detector import Detector
from typing import Optional, List, Dict, Any
import json

app = FastAPI()
detector = Detector()

# Khởi tạo ChromaFaceDB với đường dẫn mặc định

DB_PATH = "./face_db/chroma_data"
COLLECTION_NAME = "face_embeddings"
chroma_db = ChromaFaceDB(db_path=DB_PATH, collection_name=COLLECTION_NAME)


# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoint for processing frames
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...),
                        identify: bool = Query(False),
                       threshold: float = Query(0.7, ge=0.0, le=1.0),
                       top_k: int = Query(3, ge=1, le=10)):
    # Đọc nội dung file ảnh / Read image file content
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Xử lý khung hình / Process the frame
    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)
    
    # Kết quả cơ bản / Basic result
    result = {
        "persons": person_count,
        "faces": face_count,
        "person_boxes": [
            {"coords": coords, "confidence": conf}
            for (coords, conf) in person_boxes
        ],
        "face_boxes": []
    }
    
    # Xử lý thông tin khuôn mặt
    for i, face_data in enumerate(face_boxes):
        # Fix: Check the length of the tuple and handle accordingly
        if len(face_data) == 4:  # Has embedding
            coords, conf, emotion, embedding = face_data
            face_info = {
                "face_index": i,
                "coords": coords,
                "confidence": conf,
                "emotion": emotion
            }
            
            # Nếu yêu cầu nhận diện danh tính và có embedding
            if identify and embedding is not None:
                # Tìm kiếm trong database với embedding
                match = chroma_db.search_faces(np.array(embedding), top_k=top_k, threshold=threshold)
                face_info["match"] = match
                
        else:  # No embedding
            coords, conf, emotion = face_data
            face_info = {
                "face_index": i,
                "coords": coords,
                "confidence": conf,
                "emotion": emotion
            }
            
        result["face_boxes"].append(face_info)
    
    # Nếu có yêu cầu nhận diện danh tính, thêm thông tin tổng hợp
    if identify:
        identified_count = sum(1 for face in result["face_boxes"] if "match" in face and len(face["match"]) > 0)
        result["identified_count"] = identified_count
    
    return result

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ChromaDB API endpoints

@app.get("/api/database/info")
async def get_database_info():
    """Lấy thông tin tổng quan về database"""
    try:
        info = chroma_db.get_database_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin database: {str(e)}")

@app.get("/api/faces")
async def get_all_faces(limit: int = Query(100, ge=1, le=1000), 
                       offset: int = Query(0, ge=0)):
    """Lấy danh sách tất cả khuôn mặt trong database"""
    try:
        faces = chroma_db.get_all_faces(limit=limit, offset=offset)
        return {
            "total": len(faces),
            "offset": offset,
            "limit": limit,
            "faces": faces
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách khuôn mặt: {str(e)}")

@app.get("/api/faces/{face_id}")
async def get_face(face_id: str):
    """Lấy thông tin chi tiết của một khuôn mặt theo ID"""
    try:
        face = chroma_db.get_face(face_id)
        if face is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy khuôn mặt với ID: {face_id}")
        return face
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin khuôn mặt: {str(e)}")

@app.delete("/api/faces/{face_id}")
async def delete_face(face_id: str):
    """Xóa một khuôn mặt khỏi database"""
    try:
        success = chroma_db.delete_face(face_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Không thể xóa khuôn mặt với ID: {face_id}")
        return {"status": "success", "message": f"Đã xóa khuôn mặt {face_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa khuôn mặt: {str(e)}")



@app.post("/api/faces/search_embedding")
async def search_by_embedding(
    embedding: List[float],
    top_k: int = Query(5, ge=1, le=100),
    threshold: float = Query(0.7, ge=0.0, le=1.0)
):
    """Tìm kiếm khuôn mặt dựa trên embedding đã có"""
    try:
        # Chuyển list thành numpy array
        embedding_array = np.array(embedding)
        
        # Kiểm tra độ dài embedding
        if len(embedding) < 128:  # Giả sử embedding có ít nhất 128 chiều
            raise HTTPException(status_code=400, detail="Embedding không hợp lệ, quá ngắn")
        
        # Tìm kiếm trong database
        faces = chroma_db.search_faces(embedding_array, top_k=top_k, threshold=threshold)
        
        return {
            "count": len(faces),
            "matches": faces
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm với embedding: {str(e)}")

@app.post("/api/faces/register")
async def register_face(
    file: UploadFile = File(...),
    name: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    """Đăng ký khuôn mặt mới từ ảnh tải lên"""
    try:
        # Đọc ảnh và tạo embedding
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Sử dụng detector để trích xuất embedding
        embedding = detector.get_embedding_from_image(img)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh")
        
        # Tạo metadata
        metadata = {
            "name": name,
            "file_name": file.filename,
            "source": "api_upload"
        }
        
        # Thêm vào database
        face_id = chroma_db.add_face(embedding, metadata, user_id)
        
        if face_id is None:
            raise HTTPException(status_code=500, detail="Không thể đăng ký khuôn mặt")
        
        return {
            "status": "success",
            "face_id": face_id,
            "name": name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đăng ký khuôn mặt: {str(e)}")

# Mount the static files directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Serve index.html at the root
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))