import mediapipe as mp
import numpy as np
import cv2
import argparse

def get_cheek_mask(img_path):
    print(f"Processing {img_path}...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not read image {img_path}")
        return

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
        # using convexHull to ensure they are well formed polygons
        cv2.fillConvexPoly(mask, cv2.convexHull(left_pts), 255)
        cv2.fillConvexPoly(mask, cv2.convexHull(right_pts), 255)

        output_path = 'cheek_mask_result.png'
        cv2.imwrite(output_path, mask)
        print(f"Mask saved to {output_path}")

        # Optional: Save visualization
        vis = image.copy()
        vis[mask == 255] = vis[mask == 255] * 0.5 + np.array([0, 255, 0]) * 0.5
        vis_path = 'cheek_mask_vis.jpg'
        cv2.imwrite(vis_path, vis)
        print(f"Visualization saved to {vis_path}")

    else:
        print("No face detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    get_cheek_mask(args.img_path)
