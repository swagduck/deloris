# deloris_ai/rlhf_collector.py
# [MODULE: RLHF DATA COLLECTOR v1.0 - REINFORCEMENT LEARNING FROM HUMAN FEEDBACK]
# Collects (Input, Inner_Thought, Response, Score) tuples for future fine-tuning

import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import config

class RLHFDataCollector:
    """
    Hệ thống Thu thập Dữ liệu RLHF cho Deloris.
    Features:
    - Collect (Input, Inner_Thought, Response, Score) tuples
    - Store in structured format for DPO/PPO training
    - Track interaction quality over time
    - Generate training datasets
    - Analyze feedback patterns
    """
    
    def __init__(self):
        self.data_file = "data/rlhf_interactions.jsonl"
        self.training_dataset_file = "data/rlhf_training_dataset.json"
        self.current_interactions = []
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.training_dataset_file), exist_ok=True)
        
        # Statistics tracking
        self.session_stats = {
            'total_interactions': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'avg_score': 0.0,
            'session_start': time.time()
        }
        
        print("[RLHF Collector] Initialized")
    
    async def collect_interaction(self, 
                                input_text: str,
                                inner_thought: str,
                                response: str,
                                predicted_class: int,
                                upt_metrics: Dict[str, Any],
                                additional_context: Dict[str, Any] = None):
        """Collect a complete interaction for RLHF"""
        try:
            interaction_data = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'input_text': input_text,
                'inner_thought': inner_thought,
                'response': response,
                'predicted_class': predicted_class,
                'upt_metrics': upt_metrics,
                'score': None,  # Will be filled when feedback is received
                'feedback_text': None,
                'additional_context': additional_context or {},
                'session_id': self._get_session_id()
            }
            
            self.current_interactions.append(interaction_data)
            self.session_stats['total_interactions'] += 1
            
            print(f"[RLHF] Collected interaction: {input_text[:30]}...")
            
        except Exception as e:
            print(f"[RLHF] Error collecting interaction: {e}")
    
    async def save_feedback(self, score: float, feedback_text: str = None):
        """Save feedback for the most recent interaction"""
        try:
            if not self.current_interactions:
                print("[RLHF] No interactions to save feedback for")
                return
            
            # Get the most recent interaction without feedback
            for interaction in reversed(self.current_interactions):
                if interaction['score'] is None:
                    interaction['score'] = score
                    interaction['feedback_text'] = feedback_text
                    interaction['feedback_timestamp'] = time.time()
                    
                    # Update statistics
                    if score >= 0.5:
                        self.session_stats['positive_feedback'] += 1
                    else:
                        self.session_stats['negative_feedback'] += 1
                    
                    # Calculate new average score
                    scored_interactions = [i for i in self.current_interactions if i['score'] is not None]
                    if scored_interactions:
                        self.session_stats['avg_score'] = sum(i['score'] for i in scored_interactions) / len(scored_interactions)
                    
                    # Save to file
                    await self._persist_interaction(interaction)
                    
                    print(f"[RLHF] Saved feedback: score={score}, text='{feedback_text}'")
                    return
            
            print("[RLHF] No pending interactions found for feedback")
            
        except Exception as e:
            print(f"[RLHF] Error saving feedback: {e}")
    
    async def _persist_interaction(self, interaction: Dict[str, Any]):
        """Save a single interaction to the data file"""
        try:
            with open(self.data_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
            
        except Exception as e:
            print(f"[RLHF] Error persisting interaction: {e}")
    
    def _get_session_id(self) -> str:
        """Generate a session ID based on timestamp"""
        return f"session_{int(time.time())}"
    
    async def generate_training_dataset(self, min_interactions: int = 50) -> bool:
        """Generate training dataset from collected interactions"""
        try:
            # Load all interactions from file
            all_interactions = await self._load_all_interactions()
            
            if len(all_interactions) < min_interactions:
                print(f"[RLHF] Not enough interactions for training dataset ({len(all_interactions)} < {min_interactions})")
                return False
            
            # Filter interactions with feedback
            scored_interactions = [i for i in all_interactions if i.get('score') is not None]
            
            if len(scored_interactions) < min_interactions:
                print(f"[RLHF] Not enough scored interactions for training ({len(scored_interactions)} < {min_interactions})")
                return False
            
            # Create training dataset in DPO format
            training_data = []
            
            for interaction in scored_interactions:
                # Create positive and negative examples
                if interaction['score'] >= 0.7:  # High quality
                    training_data.append({
                        'prompt': interaction['input_text'],
                        'chosen': interaction['response'],
                        'rejected': self._generate_negative_example(interaction),
                        'inner_thought': interaction['inner_thought'],
                        'score': interaction['score']
                    })
                elif interaction['score'] <= 0.3:  # Low quality
                    training_data.append({
                        'prompt': interaction['input_text'],
                        'chosen': self._generate_positive_example(interaction),
                        'rejected': interaction['response'],
                        'inner_thought': interaction['inner_thought'],
                        'score': interaction['score']
                    })
            
            # Save training dataset
            with open(self.training_dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            print(f"[RLHF] Generated training dataset with {len(training_data)} examples")
            return True
            
        except Exception as e:
            print(f"[RLHF] Error generating training dataset: {e}")
            return False
    
    def _generate_negative_example(self, interaction: Dict[str, Any]) -> str:
        """Generate a negative example for DPO training"""
        # Simple negative example generation
        negative_responses = [
            "Tôi không hiểu câu hỏi của bạn.",
            "Đây là một câu trả lời không tốt.",
            "Xin lỗi, tôi không thể giúp đỡ.",
            "Câu trả lời này không liên quan."
        ]
        return negative_responses[hash(interaction['input_text']) % len(negative_responses)]
    
    def _generate_positive_example(self, interaction: Dict[str, Any]) -> str:
        """Generate a positive example for DPO training"""
        # Simple positive example generation based on good patterns
        positive_patterns = [
            "Cảm ơn bạn đã hỏi. Để tôi giúp bạn với vấn đề này.",
            "Đây là một câu hỏi thú vị. Theo tôi hiểu thì...",
            "Tôi hiểu những gì bạn đang nói. Hãy để tôi chia sẻ quan điểm của mình.",
            "Cảm ơn sự chia sẻ của bạn. Tôi nghĩ rằng..."
        ]
        return positive_patterns[hash(interaction['input_text']) % len(positive_patterns)]
    
    async def _load_all_interactions(self) -> List[Dict[str, Any]]:
        """Load all interactions from the data file"""
        try:
            if not os.path.exists(self.data_file):
                return []
            
            interactions = []
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        interactions.append(json.loads(line.strip()))
            
            return interactions
            
        except Exception as e:
            print(f"[RLHF] Error loading interactions: {e}")
            return []
    
    async def get_feedback_analysis(self) -> Dict[str, Any]:
        """Analyze feedback patterns"""
        try:
            all_interactions = await self._load_all_interactions()
            scored_interactions = [i for i in all_interactions if i.get('score') is not None]
            
            if not scored_interactions:
                return {'message': 'No scored interactions found'}
            
            # Calculate statistics
            scores = [i['score'] for i in scored_interactions]
            avg_score = sum(scores) / len(scores)
            
            # Analyze by predicted class
            class_performance = {}
            for interaction in scored_interactions:
                cls = interaction.get('predicted_class', 0)
                if cls not in class_performance:
                    class_performance[cls] = []
                class_performance[cls].append(interaction['score'])
            
            class_stats = {}
            for cls, cls_scores in class_performance.items():
                class_stats[cls] = {
                    'count': len(cls_scores),
                    'avg_score': sum(cls_scores) / len(cls_scores),
                    'min_score': min(cls_scores),
                    'max_score': max(cls_scores)
                }
            
            # Analyze by UPT metrics
            pulse_analysis = {'high': [], 'medium': [], 'low': []}
            for interaction in scored_interactions:
                pulse = interaction.get('upt_metrics', {}).get('Pulse', 0)
                if pulse > 2:
                    pulse_analysis['high'].append(interaction['score'])
                elif pulse < -2:
                    pulse_analysis['low'].append(interaction['score'])
                else:
                    pulse_analysis['medium'].append(interaction['score'])
            
            pulse_stats = {}
            for level, level_scores in pulse_analysis.items():
                if level_scores:
                    pulse_stats[level] = {
                        'count': len(level_scores),
                        'avg_score': sum(level_scores) / len(level_scores)
                    }
            
            return {
                'total_scored_interactions': len(scored_interactions),
                'overall_avg_score': avg_score,
                'class_performance': class_stats,
                'pulse_analysis': pulse_stats,
                'session_stats': self.session_stats
            }
            
        except Exception as e:
            print(f"[RLHF] Error analyzing feedback: {e}")
            return {'error': str(e)}
    
    async def export_for_training(self, format_type: str = 'dpo') -> str:
        """Export data in specific training format"""
        try:
            all_interactions = await self._load_all_interactions()
            scored_interactions = [i for i in all_interactions if i.get('score') is not None]
            
            if format_type == 'dpo':
                return await self._export_dpo_format(scored_interactions)
            elif format_type == 'ppo':
                return await self._export_ppo_format(scored_interactions)
            else:
                return "Unsupported format"
                
        except Exception as e:
            print(f"[RLHF] Error exporting data: {e}")
            return f"Error: {e}"
    
    async def _export_dpo_format(self, interactions: List[Dict[str, Any]]) -> str:
        """Export in DPO (Direct Preference Optimization) format"""
        dpo_data = []
        
        for interaction in interactions:
            if interaction['score'] >= 0.6:  # Good responses
                dpo_data.append({
                    'prompt': interaction['input_text'],
                    'chosen': interaction['response'],
                    'rejected': self._generate_negative_example(interaction)
                })
        
        filename = f"data/dpo_dataset_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        
        return f"Exported {len(dpo_data)} DPO examples to {filename}"
    
    async def _export_ppo_format(self, interactions: List[Dict[str, Any]]) -> str:
        """Export in PPO (Proximal Policy Optimization) format"""
        ppo_data = []
        
        for interaction in interactions:
            ppo_data.append({
                'prompt': interaction['input_text'],
                'response': interaction['response'],
                'reward': interaction['score'],
                'inner_thought': interaction['inner_thought']
            })
        
        filename = f"data/ppo_dataset_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ppo_data, f, ensure_ascii=False, indent=2)
        
        return f"Exported {len(ppo_data)} PPO examples to {filename}"
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.session_stats.copy()
    
    async def reset_session(self):
        """Reset session statistics"""
        self.session_stats = {
            'total_interactions': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'avg_score': 0.0,
            'session_start': time.time()
        }
        print("[RLHF] Session statistics reset")
