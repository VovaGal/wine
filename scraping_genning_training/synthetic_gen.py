import cv2
import numpy as np
import random
from pathlib import Path

class PhysicalBottleSynthesizer:
    def __init__(self, output_dir="dataset/synthetic_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def render_3d_label(self, img_path, variants=5):
        img = cv2.imread(str(img_path))
        if img is None: return
        
        # Ensure the image has an alpha (transparency) channel
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
        h, w = img.shape[:2]
        
        for i in range(variants):
            # --- 1. RANDOMIZE 3D CAMERA & BOTTLE PARAMETERS ---
            # How far the label wraps around the bottle (e.g., 120 to 160 degrees)
            fov_deg = random.uniform(120, 160)
            theta_max = np.deg2rad(fov_deg / 2.0)
            
            # Physical radius of the cylinder in pixels
            R = (w / 2.0) / theta_max
            
            # Camera angles: 
            # Pitch: Looking up/down (bends the top/bottom into ellipses)
            # Yaw: Spinning the bottle left/right
            pitch_amplitude = random.uniform(-0.15, 0.15) * h  
            yaw_offset = random.uniform(-0.3, 0.3) * theta_max 
            
            # Lighting properties (Simulating store lights / windows)
            light_x = random.uniform(-1.0, 1.0)
            shininess = random.uniform(40.0, 90.0)  # Glossiness of the paper/glass
            glare_intensity = random.uniform(120.0, 240.0) # How brightly it washes out text
            
            # --- 2. BUILD THE 3D CYLINDRICAL MESH ---
            w_out = int(2 * R * np.sin(theta_max))
            h_out = h + int(abs(pitch_amplitude))
            
            map_x = np.zeros((h_out, w_out), dtype=np.float32)
            map_y = np.zeros((h_out, w_out), dtype=np.float32)
            
            normals_x = np.zeros((h_out, w_out), dtype=np.float32)
            normals_z = np.zeros((h_out, w_out), dtype=np.float32)
            alpha_mask = np.zeros((h_out, w_out), dtype=np.float32)
            
            for x_out in range(w_out):
                # Calculate the angle (theta) on the cylinder surface for this pixel
                val = np.clip((x_out - w_out/2.0) / R, -1.0, 1.0)
                theta = np.arcsin(val) 
                
                # Apply Yaw (spin the bottle)
                theta_src = theta + yaw_offset
                
                # If rotation pushes the label behind the bottle, mask it out
                if abs(theta_src) > theta_max:
                    map_x[:, x_out] = -1 
                    alpha_mask[:, x_out] = 0
                else:
                    # Map the cylinder curve back to the flat 2D label coordinates
                    x_src = (theta_src / theta_max) * (w / 2.0) + (w / 2.0)
                    map_x[:, x_out] = x_src
                    alpha_mask[:, x_out] = 1.0
                    
                # Extract 3D surface normals (which way the bottle is facing) for lighting
                normals_x[:, x_out] = np.sin(theta)
                normals_z[:, x_out] = np.cos(theta)
                
                # Map Y pixels with a pitch curve (simulating looking down at an ellipse)
                y_curve_offset = pitch_amplitude * (np.cos(theta) - 1.0)
                
                for y_out in range(h_out):
                    y_src = y_out - max(0, pitch_amplitude) - y_curve_offset
                    map_y[y_out, x_out] = y_src

            # --- 3. EXECUTE THE 3D WARP ---
            warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            
            bgr = warped[:, :, :3].astype(np.float32)
            alpha = warped[:, :, 3].astype(np.float32) / 255.0
            
            # --- 4. CALCULATE PHYSICAL LIGHTING (Blinn-Phong Reflection) ---
            light_dir = np.array([light_x, 0.2, 0.8])
            light_dir /= np.linalg.norm(light_dir)
            view_dir = np.array([0.0, 0.0, 1.0])
            
            # Half-vector for specular light calculation
            half_vec = light_dir + view_dir
            half_vec /= np.linalg.norm(half_vec)
            
            # Diffuse Shading: Darkens the edges of the bottle curving away from the light
            diffuse = np.clip(normals_x * light_dir[0] + normals_z * light_dir[2], 0, 1)
            ambient = 0.5
            shading = ambient + (1.0 - ambient) * diffuse
            
            # Specular Glare: Creates the bright vertical reflection streak covering the text
            spec_dot = np.clip(normals_x * half_vec[0] + normals_z * half_vec[2], 0, 1)
            specular = np.power(spec_dot, shininess) * glare_intensity
            
            # --- 5. COMPOSITE IMAGE ---
            shading_3d = np.stack([shading]*3, axis=2)
            specular_3d = np.stack([specular]*3, axis=2)
            
            # Multiply flat image by shadows, then ADD the bright specular glare
            final_bgr = np.clip((bgr * shading_3d) + specular_3d, 0, 255)
            
            # Ensure background remains transparent
            final_bgra = np.concatenate([final_bgr, (alpha * alpha_mask * 255)[:, :, np.newaxis]], axis=2).astype(np.uint8)
            
            # --- 6. ADD SENSOR NOISE ---
            # Simulate slight ISO grain from a phone camera in a dimly lit store
            noise = np.random.normal(0, 4, final_bgra.shape[:2]).astype(np.float32)
            noise_3d = np.stack([noise]*3, axis=2)
            final_bgra[:, :, :3] = np.clip(final_bgra[:, :, :3].astype(np.float32) + noise_3d, 0, 255).astype(np.uint8)
            
            # Save the result
            out_path = self.output_dir / f"{img_path.stem}_3d_{i}.png"
            cv2.imwrite(str(out_path), final_bgra)
            print(f"Generated: {out_path.name}")

if __name__ == "__main__":
    synth = PhysicalBottleSynthesizer()
    for file in Path("dataset/cropped_labels").glob("*.png"):
        synth.render_3d_label(file, variants=5)