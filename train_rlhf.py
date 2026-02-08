# train_rlhf.py
# [MODULE: RLHF TRAINING PIPELINE v1.0 - DPO/PPO FINE-TUNING]
# Training pipeline for fine-tuning Deloris based on collected RLHF data

import torch
import json
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, PPOTrainer
import wandb
from typing import Dict, List, Any
import config

class RLHFTrainer:
    """
    Pipeline for training Deloris using RLHF data.
    Supports both DPO and PPO training methods.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "microsoft/DialoGPT-medium"
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
        # Training configuration
        self.output_dir = "data/rlhf_models"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[RLHF Trainer] Initialized with model: {self.model_name}")
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer for fine-tuning"""
        try:
            print("[RLHF Trainer] Loading model and tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Apply LoRA for parameter-efficient fine-tuning
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"]  # Target attention layers
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            print("[RLHF Trainer] Model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"[RLHF Trainer] Error loading model: {e}")
            raise
    
    def load_rlhf_dataset(self, dataset_path: str = "data/rlhf_training_dataset.json"):
        """Load RLHF dataset for training"""
        try:
            if not os.path.exists(dataset_path):
                print(f"[RLHF Trainer] Dataset not found: {dataset_path}")
                return False
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"[RLHF Trainer] Loaded {len(data)} training examples")
            
            # Convert to Hugging Face Dataset
            self.dataset = Dataset.from_list(data)
            
            # Preprocess dataset
            self.dataset = self.dataset.map(self._preprocess_dpo_example, batched=False)
            
            return True
            
        except Exception as e:
            print(f"[RLHF Trainer] Error loading dataset: {e}")
            return False
    
    def _preprocess_dpo_example(self, example):
        """Preprocess a single example for DPO training"""
        try:
            # Tokenize prompt
            prompt_tokens = self.tokenizer(
                example['prompt'],
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None
            )
            
            # Tokenize chosen response
            chosen_tokens = self.tokenizer(
                example['chosen'],
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors=None
            )
            
            # Tokenize rejected response
            rejected_tokens = self.tokenizer(
                example['rejected'],
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors=None
            )
            
            return {
                'prompt_input_ids': prompt_tokens['input_ids'],
                'prompt_attention_mask': prompt_tokens['attention_mask'],
                'chosen_input_ids': chosen_tokens['input_ids'],
                'chosen_attention_mask': chosen_tokens['attention_mask'],
                'rejected_input_ids': rejected_tokens['input_ids'],
                'rejected_attention_mask': rejected_tokens['attention_mask']
            }
            
        except Exception as e:
            print(f"[RLHF Trainer] Error preprocessing example: {e}")
            return None
    
    def train_dpo(self, num_epochs: int = 3, batch_size: int = 4):
        """Train using Direct Preference Optimization (DPO)"""
        try:
            if not self.model or not self.dataset:
                print("[RLHF Trainer] Model or dataset not loaded")
                return False
            
            print("[RLHF Trainer] Starting DPO training...")
            
            # Initialize wandb for tracking
            wandb.init(project="deloris-rlhf", name="dpo-training")
            
            # DPO training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                logging_steps=10,
                save_steps=500,
                eval_steps=500,
                learning_rate=5e-5,
                fp16=True,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb"
            )
            
            # Initialize DPO trainer
            dpo_trainer = DPOTrainer(
                model=self.model,
                ref_model=None,  # Use same model as reference for simplicity
                args=training_args,
                beta=0.1,  # DPO temperature parameter
                train_dataset=self.dataset,
                tokenizer=self.tokenizer,
                max_length=512,
                max_prompt_length=256
            )
            
            # Start training
            dpo_trainer.train()
            
            # Save the final model
            dpo_trainer.save_model(f"{self.output_dir}/dpo_final")
            
            print("[RLHF Trainer] DPO training completed successfully")
            wandb.finish()
            
            return True
            
        except Exception as e:
            print(f"[RLHF Trainer] Error in DPO training: {e}")
            wandb.finish()
            return False
    
    def train_ppo(self, num_epochs: int = 3, batch_size: int = 4):
        """Train using Proximal Policy Optimization (PPO)"""
        try:
            if not self.model or not self.dataset:
                print("[RLHF Trainer] Model or dataset not loaded")
                return False
            
            print("[RLHF Trainer] Starting PPO training...")
            
            # Initialize wandb for tracking
            wandb.init(project="deloris-rlhf", name="ppo-training")
            
            # PPO training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                logging_steps=10,
                save_steps=500,
                learning_rate=1.4e-5,
                fp16=True,
                save_total_limit=2,
                report_to="wandb"
            )
            
            # Reward model (simplified - using score as reward)
            def reward_model(samples):
                # Simple reward function based on response quality
                rewards = []
                for sample in samples:
                    # This is a simplified reward function
                    # In practice, you'd use a trained reward model
                    base_reward = torch.tensor(0.5)
                    rewards.append(base_reward)
                return torch.stack(rewards)
            
            # Initialize PPO trainer
            ppo_trainer = PPOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                dataset=self.dataset,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                reward_model=reward_model
            )
            
            # Start training
            ppo_trainer.train()
            
            # Save the final model
            ppo_trainer.save_model(f"{self.output_dir}/ppo_final")
            
            print("[RLHF Trainer] PPO training completed successfully")
            wandb.finish()
            
            return True
            
        except Exception as e:
            print(f"[RLHF Trainer] Error in PPO training: {e}")
            wandb.finish()
            return False
    
    def evaluate_model(self, test_dataset_path: str = "data/test_rlhf_dataset.json"):
        """Evaluate the trained model"""
        try:
            if not os.path.exists(test_dataset_path):
                print(f"[RLHF Trainer] Test dataset not found: {test_dataset_path}")
                return None
            
            with open(test_dataset_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            print(f"[RLHF Trainer] Evaluating on {len(test_data)} test examples...")
            
            # Simple evaluation - generate responses and calculate basic metrics
            total_score = 0
            for example in test_data:
                prompt = example['prompt']
                expected_response = example.get('chosen', '')
                
                # Generate response
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_length=inputs['input_ids'].shape[1] + 100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Simple similarity score (in practice, use more sophisticated metrics)
                similarity = len(set(generated_response.split()) & set(expected_response.split())) / max(len(set(generated_response.split())), 1)
                total_score += similarity
            
            avg_score = total_score / len(test_data)
            print(f"[RLHF Trainer] Average similarity score: {avg_score:.3f}")
            
            return avg_score
            
        except Exception as e:
            print(f"[RLHF Trainer] Error in evaluation: {e}")
            return None
    
    def export_for_deployment(self, model_type: str = "dpo"):
        """Export trained model for deployment"""
        try:
            model_path = f"{self.output_dir}/{model_type}_final"
            
            if not os.path.exists(model_path):
                print(f"[RLHF Trainer] Model not found: {model_path}")
                return False
            
            # Merge LoRA weights and save
            if hasattr(self.model, 'merge_and_unload'):
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(f"{self.output_dir}/{model_type}_merged")
                self.tokenizer.save_pretrained(f"{self.output_dir}/{model_type}_merged")
                
                print(f"[RLHF Trainer] Model exported for deployment: {self.output_dir}/{model_type}_merged")
            else:
                print(f"[RLHF Trainer] Model already exported: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"[RLHF Trainer] Error exporting model: {e}")
            return False

def main():
    """Main training pipeline"""
    print("=== DELORIS RLHF TRAINING PIPELINE ===")
    
    # Initialize trainer
    trainer = RLHFTrainer()
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Load RLHF dataset
    if not trainer.load_rlhf_dataset():
        print("Failed to load dataset. Please collect more data first.")
        return
    
    # Choose training method
    print("\nSelect training method:")
    print("1. DPO (Direct Preference Optimization)")
    print("2. PPO (Proximal Policy Optimization)")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        trainer.train_dpo()
        trainer.export_for_deployment("dpo")
    elif choice == "2":
        trainer.train_ppo()
        trainer.export_for_deployment("ppo")
    elif choice == "3":
        trainer.train_dpo()
        trainer.export_for_deployment("dpo")
        trainer.train_ppo()
        trainer.export_for_deployment("ppo")
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Evaluate model
    trainer.evaluate_model()
    
    print("\n=== TRAINING COMPLETED ===")
    print(f"Models saved in: {trainer.output_dir}")

if __name__ == "__main__":
    main()
