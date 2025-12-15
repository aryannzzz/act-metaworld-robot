"""
Generate evaluation videos for Standard and Modified ACT
Shows the exact failure modes visually
"""
import os
import sys
import torch
import pickle
import numpy as np
import metaworld
import argparse
import imageio
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT
from models.modified_act import ModifiedACT

# Config
TASK_CONFIG = {
    'dataset_dir': 'data/',
    'episode_len': 100,
    'state_dim': 39,
    'action_dim': 4,
    'cam_width': 480,
    'cam_height': 480,
    'camera_names': ['corner2']
}

POLICY_CONFIG = {
    'lr': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['corner2'],
}

def generate_videos(model_type, checkpoint_path, num_videos=5, output_dir='videos'):
    """Generate evaluation videos"""
    device = POLICY_CONFIG['device']
    
    print(f"\n{'='*60}")
    print(f"Generating Videos: {model_type}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Videos: {num_videos}")
    print(f"{'='*60}\n")
    
    # Load model
    if model_type == 'standard':
        model = StandardACT(
            joint_dim=TASK_CONFIG['state_dim'],
            action_dim=TASK_CONFIG['action_dim'],
            hidden_dim=POLICY_CONFIG['hidden_dim'],
            chunk_size=POLICY_CONFIG['num_queries'],
            n_heads=POLICY_CONFIG['nheads'],
            n_encoder_layers=POLICY_CONFIG['enc_layers'],
            n_decoder_layers=POLICY_CONFIG['dec_layers'],
            feedforward_dim=POLICY_CONFIG['dim_feedforward'],
            dropout=0.1,
            n_cameras=len(POLICY_CONFIG['camera_names'])
        )
    else:  # modified
        model = ModifiedACT(
            joint_dim=TASK_CONFIG['state_dim'],
            action_dim=TASK_CONFIG['action_dim'],
            hidden_dim=POLICY_CONFIG['hidden_dim'],
            chunk_size=POLICY_CONFIG['num_queries'],
            n_heads=POLICY_CONFIG['nheads'],
            n_encoder_layers=POLICY_CONFIG['enc_layers'],
            n_decoder_layers=POLICY_CONFIG['dec_layers'],
            feedforward_dim=POLICY_CONFIG['dim_feedforward'],
            dropout=0.1,
            n_cameras=len(POLICY_CONFIG['camera_names'])
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load stats
    stats_path = f'checkpoints_proper/{model_type}/norm_stats.npz'
    stats_data = np.load(stats_path)
    stats = {
        'qpos_mean': stats_data['state_mean'],
        'qpos_std': stats_data['state_std'],
        'action_mean': stats_data['action_mean'],
        'action_std': stats_data['action_std']
    }
    
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # Setup environment
    task_name = 'shelf-place-v3'
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name](render_mode='rgb_array')
    task = [task for task in ml1.train_tasks if task.env_name == task_name][0]
    env.set_task(task)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate videos
    query_frequency = POLICY_CONFIG['num_queries']
    
    for video_idx in range(num_videos):
        print(f"\nGenerating video {video_idx + 1}/{num_videos}...")
        
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        
        frames = []
        episode_info = {
            'actions_taken': [],
            'success': False,
            'episode_length': 0,
            'initial_state': obs.copy()
        }
        
        with torch.inference_mode():
            for t in range(500):
                # Render frame with annotations
                frame = env.render()
                frame = np.ascontiguousarray(frame.copy())
                
                # Add text overlay
                frame_annotated = frame.copy()
                
                # Add model info
                cv2.putText(frame_annotated, f"{model_type.upper()} ACT", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_annotated, f"Step: {t}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                frames.append(frame_annotated)
                
                # Get qpos (state)
                qpos_numpy = np.array(obs)
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                
                # Get image
                if frame.shape[:2] != (480, 480):
                    frame_resized = cv2.resize(frame, (480, 480))
                else:
                    frame_resized = frame
                
                # Prepare image tensor
                curr_image = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                
                # ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                curr_image = (curr_image.to(device) - mean) / std
                curr_image = curr_image.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
                
                # Query policy
                if t % query_frequency == 0:
                    all_actions = model(qpos, curr_image)
                
                # Select action from chunk
                raw_action = all_actions[:, t % query_frequency]
                
                # Post-process action
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = np.clip(action, -1.0, 1.0)
                
                episode_info['actions_taken'].append(action.copy())
                
                # Step environment
                step_output = env.step(action)
                if len(step_output) == 5:
                    obs, reward, terminated, truncated, info = step_output
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_output
                
                episode_info['episode_length'] += 1
                
                if done or info.get('success', False):
                    episode_info['success'] = info.get('success', False)
                    break
        
        # Add final frame with result
        final_frame = frames[-1].copy()
        result_text = "SUCCESS!" if episode_info['success'] else "FAILED"
        result_color = (0, 255, 0) if episode_info['success'] else (0, 0, 255)
        cv2.putText(final_frame, result_text, 
                   (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, result_color, 4)
        
        # Repeat final frame
        for _ in range(30):
            frames.append(final_frame)
        
        # Save video
        video_path = os.path.join(output_dir, f"{model_type}_act_episode_{video_idx+1}.mp4")
        imageio.mimsave(video_path, frames, fps=20)
        
        success_str = "✓" if episode_info['success'] else "✗"
        print(f"  {success_str} Saved: {video_path}")
        print(f"     Length: {episode_info['episode_length']} steps, Success: {episode_info['success']}")
    
    print(f"\n{'='*60}")
    print(f"All videos saved to: {output_dir}/")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['standard', 'modified', 'both'])
    parser.add_argument('--num_videos', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='videos')
    args = parser.parse_args()
    
    if args.model == 'both' or args.model == 'standard':
        checkpoint = 'checkpoints_proper/standard/best_model.pth'
        if os.path.exists(checkpoint):
            generate_videos('standard', checkpoint, args.num_videos, args.output_dir)
        else:
            print(f"Warning: {checkpoint} not found")
    
    if args.model == 'both' or args.model == 'modified':
        checkpoint = 'checkpoints_proper/modified/best_model.pth'
        if os.path.exists(checkpoint):
            generate_videos('modified', checkpoint, args.num_videos, args.output_dir)
        else:
            print(f"Warning: {checkpoint} not found")
