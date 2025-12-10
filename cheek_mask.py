import mediapipe as mp
import numpy as np
import cv2
import argparse
import os
import glob
from tqdm import tqdm

def get_cheek_mask(image):
    mp_face_mesh = mp.solutions.face_mesh
    # Initialize FaceMesh inside the function or pass it in. 
    # For batch processing, it's better to initialize once outside, but for simplicity here we can keep it lightweight or initialize once.
    # Let's initialize per call or pass it. To avoid re-init overhead, we can do it outside loop.
    pass 

def process_batch(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported extensions
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Sort to ensure consistent order across processes
    img_paths.sort()
    
    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    # --- Sharding for Distributed Processing ---
    # Get RANK and WORLD_SIZE from environment variables (defaults for single process)
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Subset the data for this process
    # e.g., if world_size=8, rank 0 gets indices 0, 8, 16... rank 1 gets 1, 9, 17...
    my_img_paths = img_paths[rank::world_size]

    print(f"Total images: {len(img_paths)}. Process {rank}/{world_size} processing {len(my_img_paths)} images.")
    # -------------------------------------------

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for img_path in tqdm(my_img_paths, desc=f"Rank {rank}"):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            res = face_mesh.process(rgb_image)
            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
                h, w, _ = image.shape
                
                # Indices for cheeks
                left_cheek_idx = [123, 116, 117, 118, 119, 120, 121, 47, 126, 209, 49, 129, 203, 206, 207, 187]
                right_cheek_idx = [352, 345, 346, 347, 348, 349, 350, 277, 355, 429, 279, 358, 423, 426, 427, 411]

                mask = np.zeros((h, w), dtype=np.uint8)

                def get_coords(indices, landmarks, width, height):
                    coords = []
                    for idx in indices:
                        pt = landmarks[idx]
                        coords.append((int(pt.x * width), int(pt.y * height)))
                    return np.array(coords, dtype=np.int32)

                left_pts = get_coords(left_cheek_idx, landmarks, w, h)
                right_pts = get_coords(right_cheek_idx, landmarks, w, h)

                # Draw filled polygons
                cv2.fillConvexPoly(mask, cv2.convexHull(left_pts), 255)
                cv2.fillConvexPoly(mask, cv2.convexHull(right_pts), 255)

                # Save mask
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(output_dir, f"{name}.png") # Saving as PNG for mask
                cv2.imwrite(output_path, mask)
            else:
                pass 
                # print(f"No face detected in {img_path}")

    if rank == 0:
        print("Batch processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output masks directory')
    args = parser.parse_args()

    process_batch(args.input_dir, args.output_dir)
