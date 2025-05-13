
import chromadb
import uuid
import numpy as np
import datetime
import logging
from pathlib import Path

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger('ChromaFaceDB')

class ChromaFaceDB:
    """
    Quản lý ChromaDB cho lưu trữ và tìm kiếm face embeddings
    """
    def __init__(self, db_path="./face_db/chroma_data", collection_name="face_embeddings"):
        """
        Khởi tạo kết nối đến ChromaDB
        
        Args:
            db_path (str): Đường dẫn đến thư mục lưu trữ ChromaDB
            collection_name (str): Tên collection lưu trữ face embeddings
        """
        # Đảm bảo thư mục tồn tại
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo client ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Lấy hoặc tạo collection
            try:
                self.collection = self.client.get_collection(collection_name)
                count = self.collection.count()
                logger.info(f"Đã kết nối đến collection {collection_name} ({count} mục)")
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Sử dụng cosine similarity
                )
                logger.info(f"Đã tạo mới collection {collection_name}")
        except Exception as e:
            logger.error(f"Lỗi khi kết nối đến ChromaDB: {e}")
            raise
    
    def add_face(self, embedding, metadata, id=None):
        """
        Thêm khuôn mặt mới vào database
        
        Args:
            embedding (np.ndarray): Vector embedding khuôn mặt
            metadata (dict): Metadata của khuôn mặt
            id (str, optional): ID của khuôn mặt. Nếu None, tự động tạo UUID
        
        Returns:
            str: ID của khuôn mặt
        """
        try:
            # Tạo ID nếu chưa có
            if id is None:
                id = str(uuid.uuid4())
                
            # Chuẩn bị dữ liệu
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Cập nhật ngày giờ
            if isinstance(metadata, dict):
                current_time = datetime.datetime.now().isoformat()
                if "created_at" not in metadata:
                    metadata["created_at"] = current_time
                metadata["updated_at"] = current_time
                
            # Thêm vào collection
            self.collection.add(
                embeddings=[embedding_list],
                metadatas=[metadata],
                ids=[id]
            )
            
            return id
        except Exception as e:
            logger.error(f"Lỗi khi thêm khuôn mặt vào database: {e}")
            return None
    
    def search_faces(self, embedding, top_k=5, threshold=0.7):
        """
        Tìm kiếm khuôn mặt tương tự trong database
        
        Args:
            embedding (np.ndarray): Vector embedding khuôn mặt cần tìm
            top_k (int): Số lượng kết quả trả về
            threshold (float): Ngưỡng tương đồng (0-1)
            
        Returns:
            list: Danh sách khuôn mặt tìm thấy với độ tương đồng cao
        """
        try:
            # Kiểm tra database trống
            if self.collection.count() == 0:
                return []
                
            # Chuyển đổi embedding sang list nếu cần
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Tìm kiếm trong collection
            results = self.collection.query(
                query_embeddings=[embedding_list],
                n_results=top_k
            )
            
            # Xử lý kết quả
            if len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
                max_sim = -1
                best_face = None

                for i in range(len(results["ids"][0])):
                    similarity = 1 - results["distances"][0][i]
                    if similarity >= threshold and similarity > max_sim:
                        max_sim = similarity
                        best_face = {
                            "id": results["ids"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity": similarity
                        }
    
                return best_face
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm khuôn mặt: {e}")
            return []
    
    def get_face(self, id):
        """
        Lấy thông tin khuôn mặt theo ID
        
        Args:
            id (str): ID của khuôn mặt
            
        Returns:
            dict: Thông tin khuôn mặt hoặc None nếu không tìm thấy
        """
        try:
            result = self.collection.get(ids=[id])
            if len(result["ids"]) > 0:
                return {
                    "id": result["ids"][0],
                    "metadata": result["metadatas"][0],
                    "embedding": result["embeddings"][0]
                }
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin khuôn mặt {id}: {e}")
            return None
    
    def update_face(self, id, embedding=None, metadata=None):
        """
        Cập nhật thông tin khuôn mặt
        
        Args:
            id (str): ID của khuôn mặt
            embedding (np.ndarray, optional): Vector embedding mới
            metadata (dict, optional): Metadata mới
            
        Returns:
            bool: True nếu cập nhật thành công, False nếu không
        """
        try:
            # Kiểm tra face có tồn tại
            face = self.get_face(id)
            if face is None:
                return False
                
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            
            if embedding is not None:
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                update_data["embeddings"] = [embedding_list]
                
            if metadata is not None:
                # Cập nhật thời gian
                metadata["updated_at"] = datetime.datetime.now().isoformat()
                update_data["metadatas"] = [metadata]
                
            # Nếu không có gì cập nhật
            if not update_data:
                return True
                
            # Thực hiện cập nhật
            self.collection.update(ids=[id], **update_data)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật khuôn mặt {id}: {e}")
            return False
    
    def delete_face(self, id):
        """
        Xóa khuôn mặt khỏi database
        
        Args:
            id (str): ID của khuôn mặt
            
        Returns:
            bool: True nếu xóa thành công, False nếu không
        """
        try:
            self.collection.delete(ids=[id])
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa khuôn mặt {id}: {e}")
            return False
    
    def get_all_faces(self, limit=1000, offset=0):
        """
        Lấy danh sách tất cả khuôn mặt trong database
        
        Args:
            limit (int): Số lượng khuôn mặt tối đa
            offset (int): Vị trí bắt đầu
            
        Returns:
            list: Danh sách khuôn mặt
        """
        try:
            result = self.collection.get(limit=limit, offset=offset)
            faces = []
            
            for i in range(len(result["ids"])):
                face_info = {
                    "id": result["ids"][i],
                    "metadata": result["metadatas"][i]
                }
                faces.append(face_info)
                
            return faces
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách khuôn mặt: {e}")
            return []
    
    def get_database_info(self):
        """
        Lấy thông tin về database
        
        Returns:
            dict: Thông tin về database
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection.name,
                "count": count,
                "status": "active" if count >= 0 else "error"
            }
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin database: {e}")
            return {"status": "error", "message": str(e)}