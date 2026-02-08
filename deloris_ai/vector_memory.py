# deloris_ai/vector_memory.py
# [MODULE: VECTOR MEMORY SYSTEM v1.0 - SEMANTIC SEARCH MEMORY]
# Replaces simple text memory with vector database for semantic search

import json
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import asyncio
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import config

class VectorMemorySystem:
    """
    Hệ thống Bộ nhớ Vector cho Deloris.
    Features:
    - Semantic search using FAISS
    - Persistent storage
    - Automatic memory consolidation
    - Relevance scoring
    - Memory aging and cleanup
    """
    
    def __init__(self, model_name=None):
        self.model_name = model_name or config.LANGUAGE_MODEL_NAME
        self.encoder = SentenceTransformer(self.model_name)
        
        # Memory storage paths
        self.index_path = config.FAISS_INDEX_CHAT_PATH
        self.metadata_path = config.FAISS_INDEX_DOCS_PATH.replace('.faiss', '_metadata.pkl')
        
        # Initialize or load existing memory
        self.index = None
        self.metadata = []
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        self._initialize_memory()
        
        # Memory management settings
        self.max_memories = 1000  # Maximum memories to keep
        self.relevance_threshold = 0.3  # Minimum similarity for relevant memories
        
    def _initialize_memory(self):
        """Initialize or load existing vector memory"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load existing memory
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"[Vector Memory] Loaded {len(self.metadata)} existing memories")
            else:
                # Create new memory
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
                self.metadata = []
                print("[Vector Memory] Created new vector memory")
                
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
                
        except Exception as e:
            print(f"[Vector Memory] Error initializing memory: {e}")
            # Fallback to empty memory
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
    
    def _normalize_vector(self, vector):
        """Normalize vector for cosine similarity"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    async def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """Add a new memory to the vector store"""
        try:
            # Generate embedding
            embedding = self.encoder.encode([text])[0]
            embedding = self._normalize_vector(embedding.reshape(1, -1))
            
            # Prepare metadata
            memory_metadata = {
                'text': text,
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'embedding': embedding.flatten().tolist(),  # Store for backup
                **(metadata or {})
            }
            
            # Add to FAISS index
            self.index.add(embedding)
            self.metadata.append(memory_metadata)
            
            # Periodic cleanup if too many memories
            if len(self.metadata) > self.max_memories:
                await self._cleanup_old_memories()
            
            print(f"[Vector Memory] Added new memory: {text[:50]}...")
            
        except Exception as e:
            print(f"[Vector Memory] Error adding memory: {e}")
    
    async def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories using semantic similarity"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.encoder.encode([query])[0]
            query_embedding = self._normalize_vector(query_embedding.reshape(1, -1))
            
            # Search in FAISS
            k = min(k, self.index.ntotal)  # Don't request more than available
            similarities, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(self.metadata):  # Valid index
                    memory = self.metadata[idx].copy()
                    memory['similarity_score'] = float(similarity)
                    memory['rank'] = i + 1
                    
                    # Only include relevant memories
                    if similarity >= self.relevance_threshold:
                        results.append(memory)
            
            return results
            
        except Exception as e:
            print(f"[Vector Memory] Error searching memories: {e}")
            return []
    
    async def get_memories_by_timeframe(self, hours_ago: int = 24) -> List[Dict[str, Any]]:
        """Get memories from a specific timeframe"""
        try:
            current_time = time.time()
            timeframe_start = current_time - (hours_ago * 3600)
            
            recent_memories = []
            for memory in self.metadata:
                if memory['timestamp'] >= timeframe_start:
                    recent_memories.append(memory.copy())
            
            # Sort by timestamp (most recent first)
            recent_memories.sort(key=lambda x: x['timestamp'], reverse=True)
            return recent_memories
            
        except Exception as e:
            print(f"[Vector Memory] Error getting timeframe memories: {e}")
            return []
    
    async def save_memory(self, chat_history: List[str], upt_state: Dict[str, Any]):
        """Save chat conversation to vector memory"""
        try:
            if not chat_history:
                return
            
            # Process chat history into meaningful memories
            for i in range(0, len(chat_history), 2):  # Process in pairs (user + response)
                if i + 1 < len(chat_history):
                    user_msg = chat_history[i]
                    deloris_msg = chat_history[i + 1]
                    
                    # Create combined memory text
                    memory_text = f"User: {user_msg}\nDeloris: {deloris_msg}"
                    
                    # Create metadata
                    metadata = {
                        'type': 'conversation',
                        'user_message': user_msg,
                        'deloris_response': deloris_msg,
                        'upt_state': upt_state,
                        'conversation_index': i // 2
                    }
                    
                    await self.add_memory(memory_text, metadata)
            
            # Also save a summary memory
            if len(chat_history) > 0:
                summary_text = f"Conversation summary: {chat_history[-4:]}"  # Last 4 messages
                summary_metadata = {
                    'type': 'summary',
                    'total_messages': len(chat_history),
                    'upt_state': upt_state
                }
                await self.add_memory(summary_text, summary_metadata)
            
            # Persist to disk
            await self._persist_memory()
            
        except Exception as e:
            print(f"[Vector Memory] Error saving conversation: {e}")
    
    async def load_memory(self) -> str:
        """Load memory summary for context"""
        try:
            # Get recent memories for context
            recent_memories = await self.get_memories_by_timeframe(hours_ago=24)
            
            if not recent_memories:
                return "Đây là phiên tương tác đầu tiên của chúng ta."
            
            # Create memory summary
            memory_summary = "=== BỘ NHỚ GẦN ĐÂY (24h) ===\n"
            
            # Add conversation memories
            conversation_memories = [m for m in recent_memories if m.get('type') == 'conversation'][:5]
            for memory in conversation_memories:
                timestamp = datetime.fromtimestamp(memory['timestamp']).strftime('%H:%M')
                memory_summary += f"[{timestamp}] {memory['user_message'][:50]}...\n"
            
            # Add summary memories
            summary_memories = [m for m in recent_memories if m.get('type') == 'summary'][:2]
            for memory in summary_memories:
                memory_summary += f"Tóm tắt: {memory['text'][:100]}...\n"
            
            memory_summary += "========================\n"
            
            return memory_summary
            
        except Exception as e:
            print(f"[Vector Memory] Error loading memory: {e}")
            return "Lỗi khi tải bộ nhớ."
    
    async def _cleanup_old_memories(self):
        """Clean up old and less relevant memories"""
        try:
            current_time = time.time()
            
            # Sort memories by relevance score and timestamp
            scored_memories = []
            for i, memory in enumerate(self.metadata):
                # Calculate aging score (older memories get lower scores)
                age_hours = (current_time - memory['timestamp']) / 3600
                age_score = max(0.1, 1.0 - (age_hours / 168))  # Decay over 1 week
                
                # Use similarity score if available, otherwise use age score
                relevance_score = memory.get('similarity_score', age_score)
                
                scored_memories.append({
                    'index': i,
                    'score': relevance_score * age_score,
                    'memory': memory
                })
            
            # Sort by score (highest first)
            scored_memories.sort(key=lambda x: x['score'], reverse=True)
            
            # Keep top memories
            memories_to_keep = scored_memories[:self.max_memories]
            
            # Rebuild index and metadata
            new_indices = [m['index'] for m in memories_to_keep]
            new_metadata = [m['memory'] for m in memories_to_keep]
            
            # Rebuild FAISS index
            new_index = faiss.IndexFlatIP(self.dimension)
            for idx in new_indices:
                if idx < len(self.metadata):
                    embedding = np.array(self.metadata[idx]['embedding']).reshape(1, -1)
                    new_index.add(embedding)
            
            self.index = new_index
            self.metadata = new_metadata
            
            print(f"[Vector Memory] Cleaned up: kept {len(new_metadata)} memories")
            
        except Exception as e:
            print(f"[Vector Memory] Error during cleanup: {e}")
    
    async def _persist_memory(self):
        """Persist memory to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            print(f"[Vector Memory] Persisted {len(self.metadata)} memories to disk")
            
        except Exception as e:
            print(f"[Vector Memory] Error persisting memory: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        try:
            current_time = time.time()
            
            # Calculate memory age distribution
            age_distribution = {'0-6h': 0, '6-24h': 0, '1-7d': 0, '7d+': 0}
            
            for memory in self.metadata:
                age_hours = (current_time - memory['timestamp']) / 3600
                if age_hours <= 6:
                    age_distribution['0-6h'] += 1
                elif age_hours <= 24:
                    age_distribution['6-24h'] += 1
                elif age_hours <= 168:
                    age_distribution['1-7d'] += 1
                else:
                    age_distribution['7d+'] += 1
            
            return {
                'total_memories': len(self.metadata),
                'index_size': self.index.ntotal,
                'dimension': self.dimension,
                'age_distribution': age_distribution,
                'model_name': self.model_name
            }
            
        except Exception as e:
            print(f"[Vector Memory] Error getting stats: {e}")
            return {}
    
    async def clear_memory(self):
        """Clear all memories"""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            
            # Delete files
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            
            print("[Vector Memory] Cleared all memories")
            
        except Exception as e:
            print(f"[Vector Memory] Error clearing memory: {e}")
