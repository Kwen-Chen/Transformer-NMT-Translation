import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

import config
from src.model import create_model
from src.dataset import DataPreprocessor, create_data_loaders
from evaluation import BLEUEvaluator


class NMTTrainer:
    """Neural Machine Translation Trainer with multi-GPU support"""
    
    def __init__(self, cfg):
        self.config = cfg
        self.device = torch.device(f'cuda:{self.config.LOCAL_RANK}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize distributed training
        if self.config.DISTRIBUTED:
            self.setup_distributed()
        
        # Setup logging
        self.setup_logging()
        
        # Load data
        self.setup_data()
        
        # Create model
        self.setup_model()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup evaluator
        self.evaluator = BLEUEvaluator()
        
        # Training state
        self.global_step = 0
        self.best_bleu = 0.0
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        self.learning_rates = []
        
    def setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.config.LOCAL_RANK)
        
    def setup_logging(self):
        """Setup logging"""
        if self.config.LOCAL_RANK == 0:
            os.makedirs(self.config.LOGS_DIR, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(self.config.LOGS_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def log(self, message):
        """Log message only from rank 0"""
        if self.logger:
            self.logger.info(message)
    
    def setup_data(self):
        """Setup data loaders"""
        self.log("Setting up data...")
        
        # Data preprocessing
        self.preprocessor = DataPreprocessor(self.config.DATA_DIR)
        
        # Try to load existing vocabularies
        if not self.preprocessor.load_vocabularies(self.config.CHECKPOINT_DIR):
            # Create new datasets and vocabularies
            train_dataset, val_dataset = self.preprocessor.create_datasets(
                max_len=self.config.MAX_LEN,
                train_ratio=0.95
            )
            self.preprocessor.save_vocabularies(self.config.CHECKPOINT_DIR)
        else:
            # Load datasets with existing vocabularies
            train_dataset, val_dataset = self.preprocessor.create_datasets(
                max_len=self.config.MAX_LEN,
                train_ratio=0.95
            )
        
        # Create data loaders with custom collate function
        self.train_loader, self.val_loader = create_data_loaders(
            train_dataset, val_dataset, 
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            distributed=self.config.DISTRIBUTED
        )
        
        self.log(f"Training batches: {len(self.train_loader)}")
        self.log(f"Validation batches: {len(self.val_loader)}")
    
    def setup_model(self):
        """Setup model"""
        self.log("Setting up model...")
        
        src_vocab_size = len(self.preprocessor.src_vocab)
        tgt_vocab_size = len(self.preprocessor.tgt_vocab)
        
        # Create model
        self.model = create_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=self.device,
            d_model=self.config.D_MODEL,
            nhead=self.config.NHEAD,
            num_encoder_layers=self.config.NUM_ENCODER_LAYERS,
            num_decoder_layers=self.config.NUM_DECODER_LAYERS,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            dropout=self.config.DROPOUT,
            max_len=self.config.MAX_LEN
        )
        
        # Setup distributed model
        if self.config.DISTRIBUTED:
            self.model = DDP(self.model, device_ids=[self.config.LOCAL_RANK])
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.preprocessor.tgt_vocab.PAD_IDX)
        
        self.log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=self.config.LOCAL_RANK != 0)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # Create masks
            model_ref = self.model.module if self.config.DISTRIBUTED else self.model
            tgt_mask = model_ref.generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
            src_key_padding_mask = model_ref.create_padding_mask(src, self.preprocessor.src_vocab.PAD_IDX)
            tgt_key_padding_mask = model_ref.create_padding_mask(tgt_input, self.preprocessor.tgt_vocab.PAD_IDX)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                src=src,
                tgt=tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Calculate loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP_GRAD)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            if self.config.LOCAL_RANK == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Log intermediate results
            if self.global_step % self.config.LOG_INTERVAL == 0 and self.config.LOCAL_RANK == 0:
                self.log(f'Step {self.global_step}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', disable=self.config.LOCAL_RANK != 0)
            
            for batch in pbar:
                # Move to device
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device) 
                tgt_output = batch['tgt_output'].to(self.device)
                
                # Create masks
                model_ref = self.model.module if self.config.DISTRIBUTED else self.model
                tgt_mask = model_ref.generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
                src_key_padding_mask = model_ref.create_padding_mask(src, self.preprocessor.src_vocab.PAD_IDX)
                tgt_key_padding_mask = model_ref.create_padding_mask(tgt_input, self.preprocessor.tgt_vocab.PAD_IDX)
                
                # Forward pass
                output = self.model(
                    src=src,
                    tgt=tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # Calculate loss
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()
                
                # Get predictions for BLEU score
                # This part needs a beam search decoder for better results.
                # For simplicity, using greedy decoding here.
                pred = output.argmax(2).transpose(0, 1) # (batch_size, seq_len)
                
                # Convert to text
                pred_text = self.preprocessor.tgt_vocab.decode_batch(pred.cpu().numpy())
                ref_text = self.preprocessor.tgt_vocab.decode_batch(batch['tgt_output'].transpose(0,1).cpu().numpy())
                
                all_predictions.extend(pred_text)
                all_references.extend([[r] for r in ref_text])
        
        avg_loss = total_loss / len(self.val_loader)
        bleu_score = self.evaluator.corpus_bleu(all_predictions, all_references)
        
        return avg_loss, bleu_score
    
    def save_checkpoint(self, epoch, bleu_score, is_best=False):
        """Save model checkpoint"""
        if self.config.LOCAL_RANK == 0:
            os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
            
            model_state = self.model.module.state_dict() if self.config.DISTRIBUTED else self.model.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'bleu_score': bleu_score,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            
            filename = os.path.join(self.config.CHECKPOINT_DIR, 'checkpoint_last.pth')
            torch.save(checkpoint, filename)
            
            if is_best:
                best_filename = os.path.join(self.config.CHECKPOINT_DIR, 'checkpoint_best.pth')
                torch.save(checkpoint, best_filename)
    
    def plot_metrics(self):
        """Plot and save training metrics"""
        if self.config.LOCAL_RANK == 0:
            plt.figure(figsize=(18, 12))
            
            # Training Loss
            plt.subplot(2, 2, 1)
            plt.plot(self.train_losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            
            # Validation Loss
            plt.subplot(2, 2, 2)
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Loss')
            plt.grid(True)
            
            # BLEU Scores
            plt.subplot(2, 2, 3)
            plt.plot(self.bleu_scores, label='BLEU-4 Score')
            plt.xlabel('Epoch')
            plt.ylabel('BLEU-4')
            plt.title('BLEU Scores')
            plt.grid(True)

            # Learning Rate
            plt.subplot(2, 2, 4)
            plt.plot(self.learning_rates, label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('LR')
            plt.title('Learning Rate')
            plt.grid(True)

            plt.tight_layout()
            save_path = os.path.join(self.config.CHECKPOINT_DIR, 'training_metrics.png')
            plt.savefig(save_path)
            self.log(f"Metrics plot saved to {save_path}")
            plt.close()

    def train(self):
        """Main training loop"""
        self.log("Starting training...")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            if self.config.DISTRIBUTED:
                self.train_loader.sampler.set_epoch(epoch)
                
            train_loss = self.train_epoch(epoch)
            val_loss, bleu_score = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.bleu_scores.append(bleu_score)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            if self.config.LOCAL_RANK == 0:
                self.log(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu_score:.4f}')
                
                is_best = bleu_score > self.best_bleu
                if is_best:
                    self.best_bleu = bleu_score
                
                self.save_checkpoint(epoch, bleu_score, is_best)
                self.plot_metrics()

            # Update scheduler
            self.scheduler.step(bleu_score)
            
        self.log("Training finished.")


def main():
    parser = argparse.ArgumentParser(description='NMT Trainer')
    
    # Distributed training args
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training. Do not set manually.')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training. Set by launch script.')
    
    args = parser.parse_args()
    
    # Update config with distributed training settings
    config.LOCAL_RANK = args.local_rank
    config.DISTRIBUTED = args.distributed or int(os.environ.get("WORLD_SIZE", 1)) > 1

    if config.DISTRIBUTED:
        config.BATCH_SIZE = config.BATCH_SIZE // int(os.environ.get("WORLD_SIZE", 1))

    trainer = NMTTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 