# training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class ACTTrainer:
    """Trainer for ACT"""
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-5),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 2000),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Loss weights
        self.beta = config.get('beta', 10.0)  # KL weight
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config.get('wandb_project', 'act-metaworld'),
                name=config.get('exp_name', 'experiment'),
                config=config
            )
    
    def compute_loss(self, pred_actions, true_actions, z_mean, z_logvar):
        """Compute CVAE loss"""
        # Reconstruction loss (L2)
        recon_loss = F.mse_loss(pred_actions, true_actions, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = kl_loss / z_mean.shape[0]  # Average over batch
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = {k: v.to(self.device) for k, v in batch['images'].items()}
            joints = batch['joints'].to(self.device)
            actions = batch['actions'].to(self.device)
            
            # Forward pass
            pred_actions, z_mean, z_logvar = self.model(
                images, joints, actions, training=True
            )
            
            # Compute loss
            loss, recon_loss, kl_loss = self.compute_loss(
                pred_actions, actions, z_mean, z_logvar
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}"
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        return avg_loss, avg_recon, avg_kl
    
    def validate(self, dataloader):
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = {k: v.to(self.device) for k, v in batch['images'].items()}
                joints = batch['joints'].to(self.device)
                actions = batch['actions'].to(self.device)
                
                pred_actions, z_mean, z_logvar = self.model(
                    images, joints, actions, training=True
                )
                
                loss, recon_loss, kl_loss = self.compute_loss(
                    pred_actions, actions, z_mean, z_logvar
                )
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        return avg_loss, avg_recon, avg_kl
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader)
            
            # Step scheduler
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/recon': train_recon,
                    'train/kl': train_kl,
                    'val/loss': val_loss,
                    'val/recon': val_recon,
                    'val/kl': val_kl,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('checkpoints/best.pth', epoch)
            
            # Save periodic checkpoint
            if epoch % self.config.get('save_freq', 100) == 0:
                self.save_checkpoint(f'checkpoints/epoch_{epoch}.pth', epoch)
    
    def save_checkpoint(self, filename, epoch):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, filename)
        print(f"✓ Saved checkpoint: {filename}")
