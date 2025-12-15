"""
Create demonstration videos showing the failure mode
Records episodes from both Standard and Modified ACT models
"""
import os
import sys
import torch
import numpy as np
import metaworld
import imageio
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.standard_act import StandardACT
from models.modified_act import ModifiedACT

def create_demo_videos():
    """Create simple demo videos showing model behavior"""
    
    task_name = 'shelf-place-v3'
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name](render_mode='rgb_array')
    task = [task for task in ml1.train_tasks if task.env_name == task_name][0]
    
    os.makedirs('videos', exist_ok=True)
    
    # Create videos for each model type
    for model_type in ['standard', 'modified']:
        print(f"\n{'='*60}")
        print(f"Creating demo video for {model_type.upper()} ACT")
        print(f"{'='*60}\n")
        
        # Load normalization stats
        stats_data = np.load(f'checkpoints_proper/{model_type}/norm_stats.npz')
        state_mean = stats_data['state_mean']
        state_std = stats_data['state_std']
        action_mean = stats_data['action_mean']
        action_std = stats_data['action_std']
        
        # Load model
        if model_type == 'standard':
            model = StandardACT(
                joint_dim=39, action_dim=4, hidden_dim=512,
                chunk_size=100, n_heads=8, n_encoder_layers=4,
                n_decoder_layers=7, feedforward_dim=3200,
                dropout=0.1, n_cameras=1
            )
        else:
            model = ModifiedACT(
                joint_dim=39, action_dim=4, hidden_dim=512,
                chunk_size=100, n_heads=8, n_encoder_layers=4,
                n_decoder_layers=7, feedforward_dim=3200,
                dropout=0.1, n_cameras=1
            )
        
        checkpoint = torch.load(
            f'checkpoints_proper/{model_type}/best_model.pth',
            map_location='cuda',
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        model.eval()
        
        # Record 3 episodes
        for ep_idx in range(3):
            print(f"Recording episode {ep_idx + 1}/3...")
            
            env.set_task(task)
            reset_output = env.reset()
            obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            
            frames = []
            
            with torch.inference_mode():
                for t in range(500):
                    # Render
                    frame = env.render()
                    
                    # Add overlay
                    frame_copy = frame.copy()
                    cv2.putText(frame_copy, f"{model_type.upper()} ACT", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_copy, f"Step: {t}/500", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    frames.append(frame_copy)
                    
                    # Get normalized state
                    state = (obs - state_mean) / state_std
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).cuda()
                    
                    # Get image
                    if frame.shape[:2] != (480, 480):
                        img = cv2.resize(frame, (480, 480))
                    else:
                        img = frame.copy()
                    
                    # Normalize image
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
                    img_tensor = (img_tensor.cuda() - mean) / std
                    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
                    
                    # Get action (query every 100 steps)
                    if t % 100 == 0:
                        all_actions = model(state_tensor, img_tensor)
                    
                    # Use action from current chunk position
                    action_norm = all_actions[0, t % 100].cpu().numpy()
                    action = action_norm * action_std + action_mean
                    action = np.clip(action, -1.0, 1.0)
                    
                    # Step
                    step_output = env.step(action)
                    if len(step_output) == 5:
                        obs, reward, terminated, truncated, info = step_output
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_output
                    
                    if done or info.get('success', False):
                        break
            
            # Add result overlay
            final_frame = frames[-1].copy()
            result_text = "SUCCESS" if info.get('success', False) else "FAILED"
            color = (0, 255, 0) if info.get('success', False) else (0, 0, 255)
            cv2.putText(final_frame, result_text, 
                       (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            
            # Hold final frame
            for _ in range(30):
                frames.append(final_frame)
            
            # Save video
            video_path = f'videos/{model_type}_act_demo_{ep_idx+1}.mp4'
            imageio.mimsave(video_path, frames, fps=20)
            print(f"  ✓ Saved: {video_path}")
    
    print(f"\n{'='*60}")
    print("✅ All demo videos created!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    create_demo_videos()
