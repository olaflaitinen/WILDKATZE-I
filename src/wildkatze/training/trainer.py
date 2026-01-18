"""
WILDKATZE-I Training Implementation

This module implements the training pipeline for WILDKATZE-I, including:
- Distributed training with DeepSpeed/FSDP support
- Gradient checkpointing for memory efficiency
- Mixed precision training (BFloat16)
- Checkpoint management
- Logging and monitoring

The trainer supports pretraining, fine-tuning, and RLHF workflows.

Copyright (c) 2026 olaflaitinen. All rights reserved.
Licensed under EUPL v1.2
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import math
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    """
    Arguments for the training process.
    
    Attributes:
        output_dir: Directory for saving checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Steps to accumulate before update
        learning_rate: Peak learning rate
        weight_decay: L2 regularization coefficient
        warmup_ratio: Ratio of warmup steps
        lr_scheduler_type: Type of learning rate scheduler
        bf16: Whether to use BFloat16 precision
        gradient_checkpointing: Whether to use gradient checkpointing
        logging_steps: Steps between logging
        save_steps: Steps between checkpoints
        save_total_limit: Maximum checkpoints to keep
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
    """
    output_dir: str = "./models/checkpoints"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    max_grad_norm: float = 1.0
    seed: int = 42
    dataloader_num_workers: int = 4
    resume_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Validate arguments after initialization."""
        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must be between 0 and 1")


class TrainingState:
    """
    Holds the state of the training process.
    
    Attributes:
        global_step: Total number of optimization steps
        epoch: Current epoch
        best_loss: Best validation loss seen
        total_tokens: Total tokens processed
    """
    
    def __init__(self):
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.total_tokens = 0
        self.log_history = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "total_tokens": self.total_tokens,
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "TrainingState":
        """Create state from dictionary."""
        state = cls()
        state.global_step = state_dict.get("global_step", 0)
        state.epoch = state_dict.get("epoch", 0)
        state.best_loss = state_dict.get("best_loss", float('inf'))
        state.total_tokens = state_dict.get("total_tokens", 0)
        return state


class WildkatzeTrainer:
    """
    Trainer class for WILDKATZE-I models.
    
    This trainer handles the complete training workflow including:
    - Optimizer and scheduler setup
    - Gradient accumulation and mixed precision
    - Checkpointing and resumption
    - Logging and monitoring
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataloader: DataLoader for training data
        eval_dataloader: Optional DataLoader for evaluation
        optimizers: Optional tuple of (optimizer, scheduler)
        callbacks: Optional list of callback functions
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizers: Optional[tuple] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        if optimizers is not None:
            self.optimizer, self.scheduler = optimizers
        else:
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
        # Setup mixed precision
        self.scaler = GradScaler() if args.fp16 else None
        
        # Training state
        self.state = TrainingState()
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing'):
                self.model.model.gradient_checkpointing = True
                
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Training arguments: {args}")
        
    def _create_optimizer(self) -> AdamW:
        """
        Create the AdamW optimizer with weight decay.
        
        Applies weight decay to all parameters except biases and
        layer normalization weights.
        """
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        no_decay_patterns = ["bias", "layernorm", "layer_norm", "ln_", "norm"]
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(pattern in name.lower() for pattern in no_decay_patterns):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create the learning rate scheduler."""
        num_training_steps = (
            len(self.train_dataloader) 
            * self.args.num_train_epochs 
            // self.args.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        
        if self.args.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
            )
        elif self.args.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps,
            )
        else:
            scheduler = None
            
        return scheduler
    
    def _training_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing input_ids, attention_mask, labels
            
        Returns:
            Loss tensor
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        with autocast(enabled=self.args.bf16, dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels", batch["input_ids"]),
            )
            
            if isinstance(outputs, tuple):
                logits, loss = outputs
            else:
                loss = outputs.loss
                
        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps
        
        return loss
    
    def _backward_step(self, loss: torch.Tensor):
        """
        Perform backward pass.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.args.max_grad_norm
        )
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        if self.scheduler is not None:
            self.scheduler.step()
            
        self.optimizer.zero_grad()
        
    def train(self) -> Dict[str, float]:
        """
        Run the training loop.
        
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        if self.args.resume_from_checkpoint:
            self._load_checkpoint(self.args.resume_from_checkpoint)
            
        total_steps = (
            len(self.train_dataloader) 
            * self.args.num_train_epochs 
            // self.args.gradient_accumulation_steps
        )
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Total epochs: {self.args.num_train_epochs}")
        
        self.model.train()
        accumulated_loss = 0.0
        
        for epoch in range(self.state.epoch, self.args.num_train_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                # Training step
                loss = self._training_step(batch)
                accumulated_loss += loss.item()
                epoch_loss += loss.item() * self.args.gradient_accumulation_steps
                
                # Backward pass
                self._backward_step(loss)
                
                # Track tokens
                batch_tokens = batch["input_ids"].numel()
                self.state.total_tokens += batch_tokens
                
                # Optimizer step
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    self.state.global_step += 1
                    
                    # Logging
                    if self.state.global_step % self.args.logging_steps == 0:
                        avg_loss = accumulated_loss / self.args.gradient_accumulation_steps
                        lr = self.optimizer.param_groups[0]["lr"]
                        
                        log_entry = {
                            "step": self.state.global_step,
                            "loss": avg_loss,
                            "lr": lr,
                            "epoch": epoch,
                            "tokens": self.state.total_tokens,
                        }
                        self.state.log_history.append(log_entry)
                        
                        logger.info(
                            f"Step {self.state.global_step}: "
                            f"loss={avg_loss:.4f}, lr={lr:.2e}"
                        )
                        
                        accumulated_loss = 0.0
                        
                    # Checkpointing
                    if self.state.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()
                        
                    # Run callbacks
                    for callback in self.callbacks:
                        callback(self, self.state, log_entry)
                        
            # End of epoch evaluation
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                logger.info(f"Epoch {epoch} evaluation loss: {eval_loss:.4f}")
                
                if eval_loss < self.state.best_loss:
                    self.state.best_loss = eval_loss
                    self._save_checkpoint(is_best=True)
                    
        # Final save
        self._save_checkpoint()
        
        return {
            "train_loss": epoch_loss / len(self.train_dataloader),
            "total_steps": self.state.global_step,
            "total_tokens": self.state.total_tokens,
        }
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run evaluation loop.
        
        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast(enabled=self.args.bf16, dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels", batch["input_ids"]),
                )
                
                if isinstance(outputs, tuple):
                    _, loss = outputs
                else:
                    loss = outputs.loss
                    
            total_loss += loss.item()
            total_steps += 1
            
        self.model.train()
        return total_loss / total_steps
    
    def _save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint
        """
        checkpoint_dir = Path(self.args.output_dir)
        
        if is_best:
            save_path = checkpoint_dir / "best_model"
        else:
            save_path = checkpoint_dir / f"checkpoint-{self.state.global_step}"
            
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(
            self.model.state_dict(),
            save_path / "pytorch_model.bin"
        )
        
        # Save optimizer and scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }, save_path / "training_state.bin")
        
        # Save trainer state
        with open(save_path / "trainer_state.json", "w") as f:
            json.dump(self.state.to_dict(), f)
            
        logger.info(f"Checkpoint saved to {save_path}")
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
        
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        self.model.load_state_dict(
            torch.load(checkpoint_dir / "pytorch_model.bin", map_location=self.device)
        )
        
        # Load optimizer and scheduler
        training_state = torch.load(
            checkpoint_dir / "training_state.bin", 
            map_location=self.device
        )
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self.scheduler and training_state["scheduler"]:
            self.scheduler.load_state_dict(training_state["scheduler"])
        if self.scaler and training_state["scaler"]:
            self.scaler.load_state_dict(training_state["scaler"])
            
        # Load trainer state
        with open(checkpoint_dir / "trainer_state.json") as f:
            self.state = TrainingState.from_dict(json.load(f))
            
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only save_total_limit."""
        if self.args.save_total_limit is None:
            return
            
        checkpoint_dir = Path(self.args.output_dir)
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        while len(checkpoints) > self.args.save_total_limit:
            oldest = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {oldest}")
            import shutil
            shutil.rmtree(oldest)
