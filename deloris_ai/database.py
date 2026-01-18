# deloris_ai/database.py
import sqlite3
import json
import time
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'deloris_memory.db')

class DelorisDB:
    def __init__(self):
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        """Khởi tạo cấu trúc bảng nếu chưa tồn tại"""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 1. Bảng lưu lịch sử tìm kiếm (Thay cho oracle_history.log)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                results TEXT,
                timestamp REAL
            )
        ''')

        # 2. Bảng lưu phản hồi nóng (Thay cho active_feedback_training.jsonl)
        # Cột 'processed' để đánh dấu xem Dreaming module đã xử lý chưa
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                model_output TEXT,
                upt_state TEXT, 
                rating INTEGER,
                processed INTEGER DEFAULT 0,
                timestamp REAL
            )
        ''')

        # 3. Bảng dữ liệu huấn luyện (Thay cho training_dataset.json)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input TEXT,
                output TEXT,
                upt_context TEXT,
                source TEXT DEFAULT 'feedback', 
                created_at REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    # --- CÁC HÀM CHO ORACLE ---
    def log_search(self, query, results):
        conn = self._get_conn()
        conn.execute('INSERT INTO search_logs (query, results, timestamp) VALUES (?, ?, ?)',
                     (query, results, time.time()))
        conn.commit()
        conn.close()

    # --- CÁC HÀM CHO PLASTICITY (HỌC TẬP) ---
    def log_feedback(self, user_input, model_output, upt_state, rating):
        conn = self._get_conn()
        conn.execute('''
            INSERT INTO feedback_logs (user_input, model_output, upt_state, rating, timestamp) 
            VALUES (?, ?, ?, ?, ?)
        ''', (user_input, model_output, json.dumps(upt_state), rating, time.time()))
        conn.commit()
        conn.close()

    # --- CÁC HÀM CHO DREAMING (TỔNG HỢP KÝ ỨC) ---
    def fetch_unprocessed_feedback(self):
        """Lấy các phản hồi tốt (rating > 0) chưa được xử lý"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT * FROM feedback_logs WHERE processed = 0 AND rating > 0')
        rows = cursor.fetchall()
        
        data = []
        ids = []
        for row in rows:
            data.append({
                "input": row['user_input'],
                "output": row['model_output'],
                "upt_context": json.loads(row['upt_state'])
            })
            ids.append(row['id'])
        
        conn.close()
        return data, ids

    def mark_feedback_processed(self, ids):
        if not ids: return
        conn = self._get_conn()
        placeholders = ', '.join(['?'] * len(ids))
        conn.execute(f'UPDATE feedback_logs SET processed = 1 WHERE id IN ({placeholders})', ids)
        conn.commit()
        conn.close()

    def add_training_samples(self, samples, source='dream'):
        """Thêm mẫu mới vào kho tri thức vĩnh cửu"""
        conn = self._get_conn()
        for s in samples:
            conn.execute('''
                INSERT INTO training_data (input, output, upt_context, source, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (s['input'], s['output'], json.dumps(s['upt_context']), source, time.time()))
        conn.commit()
        conn.close()

    def get_all_training_data(self):
        """Lấy toàn bộ dữ liệu để train lại model"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT input, output, upt_context FROM training_data')
        
        dataset = []
        for row in cursor:
            dataset.append({
                "input": row['input'],
                "output": row['output'],
                "upt_context": json.loads(row['upt_context'])
            })
        conn.close()
        return dataset