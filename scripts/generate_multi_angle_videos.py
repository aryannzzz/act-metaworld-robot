"""
Generate evaluation videos from multiple camera angles to better visualize model performance.
"""

import os
import sys
import torch
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.standard_act import StandardACT
from models.modified_act import ModifiedACT
from envs.metaworld_simple_wrapper import SimpleMetaWorldWrapper


def record_multi_angle_episode(env, model, checkpoint_path, model_name, device, 
                                cameras=['corner2', 'topview', 'behindGripper'],
                                max_steps=500):
    """
    Record a single episode from multiple camera angles.
    
    Returns:
        camera_frames: dict mapping camera_name -> list of frames
        success: whether the episode succeeded
        diagnostics: dict with detailed information
    """
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Reset environment
    obs = env.reset()
    
    # Initialize frame storage for each camera
    camera_frames = {cam: [] for cam in cameras}
    
    success = False
    episode_rewards = []
    success_values = []
    object_positions = []
    gripper_positions = []
    
    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            # Capture frames from all cameras
            for camera_name in cameras:
                try:
                    # Access the base MetaWorld environment
                    base_env = env.env
                    
                    # Set camera and render
                    base_env.mujoco_renderer._camera_name = camera_name
                    frame = base_env.render()
                    
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        frame = (frame * 255).astype(np.uint8)
                    
                    camera_frames[camera_name].append(frame)
                except Exception as e:
                    print(f"Warning: Could not capture {camera_name}: {e}")
                    if len(camera_frames[camera_name]) > 0:
                        # Reuse last frame
                        camera_frames[camera_name].append(camera_frames[camera_name][-1])
                    else:
                        # Create blank frame
                        camera_frames[camera_name].append(np.zeros((480, 480, 3), dtype=np.uint8))
            
            # Get model prediction
            joints = obs['joints'] if 'joints' in obs else obs['state']
            joints_tensor = torch.from_numpy(joints).float().unsqueeze(0).to(device)
            
            # Prepare images - wrapper already provides normalized CHW format
            img_obs = obs['images']
            if isinstance(img_obs, dict):
                # Use one of the camera views
                img = list(img_obs.values())[0]  # Already CHW, normalized
            else:
                img = img_obs
            
            # Convert to tensor and add batch dimension
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            images_dict = {'default': img_tensor}
            
            # Predict action chunk
            action_chunk = model(images_dict, joints_tensor, training=False)
            action_chunk = action_chunk.squeeze(0).cpu().numpy()  # [chunk_size, action_dim]
            
            # Use first action from chunk
            action = action_chunk[0]
            action = np.clip(action, -1, 1)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Collect diagnostics
            episode_rewards.append(reward)
            success_values.append(info.get('success', 0.0))
            
            # Try to get object and gripper positions
            try:
                base_env = env.env
                # Object position (puck)
                obj_pos = base_env.get_body_com("obj")[:3]
                object_positions.append(obj_pos.copy())
                
                # Gripper position
                gripper_pos = base_env.get_body_com("hand")[:3]
                gripper_positions.append(gripper_pos.copy())
            except:
                pass
            
            if info.get('success', 0.0) > 0:
                success = True
                print(f"  ✓ Success detected at step {step}!")
                break
            
            if done:
                break
    
    # Compile diagnostics
    diagnostics = {
        'num_steps': step + 1,
        'success': success,
        'total_reward': sum(episode_rewards),
        'avg_reward': np.mean(episode_rewards),
        'max_success_value': max(success_values) if success_values else 0.0,
        'final_success_value': success_values[-1] if success_values else 0.0,
        'object_positions': object_positions,
        'gripper_positions': gripper_positions,
    }
    
    # Add goal distance if we have object positions
    if object_positions and hasattr(env.env, '_target_pos'):
        goal_pos = env.env._target_pos
        final_obj_pos = object_positions[-1]
        distance_to_goal = np.linalg.norm(final_obj_pos - goal_pos)
        diagnostics['distance_to_goal'] = distance_to_goal
        diagnostics['goal_position'] = goal_pos
        diagnostics['final_object_position'] = final_obj_pos
        print(f"  Distance to goal: {distance_to_goal:.4f}m")
        print(f"  Goal: {goal_pos}, Final obj: {final_obj_pos}")
    
    return camera_frames, success, diagnostics


def create_side_by_side_video(camera_frames_dict, output_path, fps=10):
    """
    Create a video with multiple camera views arranged side by side.
    
    Args:
        camera_frames_dict: dict mapping camera_name -> list of frames
        output_path: where to save the video
        fps: frames per second
    """
    # Get camera names and verify all have same length
    cameras = list(camera_frames_dict.keys())
    num_frames = len(camera_frames_dict[cameras[0]])
    
    print(f"  Creating {len(cameras)}-camera video with {num_frames} frames...")
    
    # Create combined frames
    combined_frames = []
    for i in range(num_frames):
        # Get frame from each camera
        frames = [camera_frames_dict[cam][i] for cam in cameras]
        
        # Arrange horizontally (side by side)
        if len(frames) == 3:
            # 3 cameras: arrange in a row
            combined = np.hstack(frames)
        elif len(frames) == 2:
            # 2 cameras: side by side
            combined = np.hstack(frames)
        else:
            # Single camera
            combined = frames[0]
        
        combined_frames.append(combined)
    
    # Save video
    imageio.mimsave(output_path, combined_frames, fps=fps)
    print(f"  ✓ Saved to {output_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Paths
    standard_checkpoint = 'experiments/standard_act_20251212_185014/checkpoints/best.pth'
    modified_checkpoint = 'experiments/modified_act_20251212_225539/checkpoints/best.pth'
    output_dir = Path('evaluation_videos_multi_angle')
    output_dir.mkdir(exist_ok=True)
    
    # Camera angles to use
    cameras = ['topview', 'corner2', 'behindGripper']
    
    # Test both models
    models_to_test = [
        ('StandardACT', standard_checkpoint, StandardACT),
        ('ModifiedACT', modified_checkpoint, ModifiedACT),
    ]
    
    for model_name, checkpoint_path, model_class in models_to_test:
        print(f"\n{'='*70}")
        print(f"Recording {model_name}")
        print('='*70)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        # Create model
        if model_class == StandardACT:
            model = StandardACT(
                joint_dim=config['model']['joint_dim'],
                action_dim=config['model']['action_dim'],
                hidden_dim=config['model']['hidden_dim'],
                latent_dim=config['model']['latent_dim'],
                n_encoder_layers=config['model']['n_encoder_layers'],
                n_decoder_layers=config['model']['n_decoder_layers'],
                n_heads=config['model']['n_heads'],
                feedforward_dim=config['model']['feedforward_dim'],
                chunk_size=config['chunking']['chunk_size'],
                dropout=config['model']['dropout']
            )
        else:  # ModifiedACT
            model = ModifiedACT(
                joint_dim=config['model']['joint_dim'],
                action_dim=config['model']['action_dim'],
                hidden_dim=config['model']['hidden_dim'],
                latent_dim=config['model']['latent_dim'],
                n_encoder_layers=config['model']['n_encoder_layers'],
                n_decoder_layers=config['model']['n_decoder_layers'],
                n_heads=config['model']['n_heads'],
                feedforward_dim=config['model']['feedforward_dim'],
                chunk_size=config['chunking']['chunk_size'],
                image_size=tuple(config['env']['image_size']),
                dropout=config['model']['dropout']
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Create environment
        env = SimpleMetaWorldWrapper('shelf-place-v3')
        
        print(f"Recording episode from {len(cameras)} angles: {cameras}")
        
        # Record episode
        camera_frames, success, diagnostics = record_multi_angle_episode(
            env, model, checkpoint_path, model_name, device, cameras=cameras
        )
        
        # Print diagnostics
        print(f"\nDiagnostics:")
        print(f"  Steps: {diagnostics['num_steps']}")
        print(f"  Success: {'✓ YES' if diagnostics['success'] else '✗ NO'}")
        print(f"  Total reward: {diagnostics['total_reward']:.3f}")
        print(f"  Avg reward: {diagnostics['avg_reward']:.4f}")
        print(f"  Max success value: {diagnostics['max_success_value']:.3f}")
        print(f"  Final success value: {diagnostics['final_success_value']:.3f}")
        if 'distance_to_goal' in diagnostics:
            print(f"  Final distance to goal: {diagnostics['distance_to_goal']:.4f}m")
        
        # Create video with all cameras side by side
        video_path = output_dir / f"{model_name.lower()}_multi_angle.mp4"
        create_side_by_side_video(camera_frames, video_path, fps=20)
        
        # Also save individual camera videos
        for cam_name, frames in camera_frames.items():
            individual_path = output_dir / f"{model_name.lower()}_{cam_name}.mp4"
            imageio.mimsave(individual_path, frames, fps=20)
            print(f"  ✓ Saved {cam_name} view to {individual_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ All videos saved to {output_dir}/")
    print('='*70)


if __name__ == '__main__':
    main()
