"""
Policy wrapper adapted from ACT-original/training/policy.py
"""
import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT

def kl_divergence(mu, logvar):
    """KL divergence computation"""
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    
    return total_kld, dimension_wise_kld, mean_kld

class ACTPolicy(nn.Module):
    """ACT Policy wrapper matching original implementation"""
    def __init__(self, args_override):
        super().__init__()
        # Create the model
        self.model = StandardACT(
            joint_dim=39,
            action_dim=4,
            chunk_size=args_override['num_queries'],
            hidden_dim=args_override['hidden_dim'],
            feedforward_dim=args_override['dim_feedforward'],
            n_encoder_layers=args_override['enc_layers'],
            n_decoder_layers=args_override['dec_layers'],
            n_heads=args_override['nheads'],
            n_cameras=len(args_override['camera_names'])
        )
        self.kl_weight = args_override['kl_weight']
        self.lr = args_override['lr']
        self.lr_backbone = args_override['lr_backbone']
        self.num_queries = args_override['num_queries']
        print(f'KL Weight: {self.kl_weight}')
    
    def __call__(self, qpos, image, actions=None, is_pad=None):
        """Forward pass matching original ACT"""
        # Handle single camera case: [B, 1, C, H, W] -> [B, C, H, W]
        if image.dim() == 5 and image.shape[1] == 1:
            image = image.squeeze(1)
        
        # ImageNet normalization (CRITICAL!)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        
        if actions is not None:  # Training time
            actions = actions[:, :self.num_queries]
            is_pad = is_pad[:, :self.num_queries]
            
            # Prepare images dict
            images_dict = {'corner2': image}
            
            # Forward through model (pass actions for VAE encoder)
            pred_actions, mu, logvar = self.model(images_dict, qpos, actions=actions, training=True)
            
            # Compute losses
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, pred_actions, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # Inference time
            images_dict = {'corner2': image}
            pred_actions = self.model(images_dict, qpos, training=False)
            return pred_actions
    
    def configure_optimizers(self):
        """Configure optimizer like original"""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
