import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch'))
from model import BiSeNet

import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse
import tqdm

parser = argparse.ArgumentParser()
# Input/output args (support aliases for convenience)
parser.add_argument('--data', '--input', '--input_dir', dest='data', required=True, help='Path to input images directory')
parser.add_argument('--output_path', '--output', '--output_dir', dest='output_path', required=True, help='Directory to write results')
# Pretrained model/checkpoint path
parser.add_argument('--model', '--checkpoint', '--cp', dest='cp', required=True, help='Path to pretrained model .pth file')
parser.add_argument('--parts', dest='parts', default='all', help='Comma-separated list of parts to extract (e.g., "mouth,eyes"). Default: all')
args = parser.parse_args()

def process_eyes(eyes_uint8, face):
    """Process eye regions by extending bounding box."""
    contours, _ = cv2.findContours(eyes_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure there are valid contours before proceeding
    if not contours:
        return eyes_uint8

    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    extended_h = int(h * 1.3)
    extended_y_bottom = min(y + extended_h, face.shape[0])

    # Iterate safely within array bounds
    for i in range(max(0, x), min(x + w, face.shape[1])):  
        for j in range(max(0, y), extended_y_bottom):
            if face[j, i]: 
                eyes_uint8[j, i] = 255

    return eyes_uint8


def get_cheek_mask(skin_mask, eye_mask, nose_mask, face_side='left'):
    """
    Generates a single-side cheek mask based on relative facial feature positions.
    
    Args:
        skin_mask (np.array): Binary mask of the skin (uint8).
        eye_mask (np.array): Binary mask of the eye + eyebrow on the corresponding side (uint8).
        nose_mask (np.array): Binary mask of the nose (uint8).
        face_side (str): 'left' for image left (subject's right cheek), 'right' for image right (subject's left cheek).
    
    Returns:
        np.array: Binary mask of the cheek area (uint8).
    """
    # 1. Get Bounding Boxes for features
    contours_eye, _ = cv2.findContours(eye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_nose, _ = cv2.findContours(nose_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fail-safe: Return empty mask if features are missing (e.g., extreme profile view)
    if not contours_eye or not contours_nose:
        return np.zeros_like(skin_mask)

    e_x, e_y, e_w, e_h = cv2.boundingRect(np.vstack(contours_eye))
    n_x, n_y, n_w, n_h = cv2.boundingRect(np.vstack(contours_nose))
    
    roi = np.zeros_like(skin_mask)
    h, w = roi.shape
    
    # 2. Define Geometric Boundaries (Heuristics)
    # Top: Bottom edge of the eye BBox
    top = e_y + e_h
    # Bottom: Bottom edge of the nose BBox (scaled by 0.8 to avoid the jawline)
    bottom = n_y + int(n_h * 0.8)
    
    # Boundary safety checks
    top = max(0, top)
    bottom = min(h, bottom)
    if top >= bottom: return np.zeros_like(skin_mask)

    # 3. Define Horizontal Boundaries
    if face_side == 'left':
        # Left Cheek (Image Left): From image left edge/eye corner to nose left edge
        right_bound = n_x
        # Extend slightly outwards to capture the area under the outer eye corner
        left_bound = max(0, e_x - 20) 
        roi[top:bottom, left_bound:right_bound] = 255
    else:
        # Right Cheek (Image Right): From nose right edge to image right edge
        left_bound = n_x + n_w
        right_bound = min(w, e_x + e_w + 20)
        roi[top:bottom, left_bound:right_bound] = 255
        
    # 4. Core Operation: Intersect ROI with Semantic Skin Mask
    cheek_final = cv2.bitwise_and(roi, skin_mask)
    
    return cheek_final



def vis_parsing_maps(im, parsing_anno, stride, parts, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    # atts = [0 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
    # 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
    
    # We only need to save specific masks, so we don't need the full visualization logic here
    # but we keep the structure consistent with inference.py for simplicity

    if save_im:
        im_name = os.path.splitext(os.path.basename(save_path))[0]
        base_dir = os.path.dirname(save_path)
        # We want to save directly to subfolders in the output path, so we use base_dir which is passed as respth
        
        if 'all' in parts:
            parts = ['mouth', 'brows', 'eyes', 'cheeks']

        skin = (parsing_anno == 1)[..., None]
        l_brow = (parsing_anno == 2)[..., None]
        r_brow = (parsing_anno == 3)[..., None]
        l_eye = (parsing_anno == 4)[..., None]
        r_eye = (parsing_anno == 5)[..., None]
        eye_g = (parsing_anno == 6)[..., None]
        l_ear = (parsing_anno == 7)[..., None]
        r_ear = (parsing_anno == 8)[..., None]
        ear_r = (parsing_anno == 9)[..., None]
        nose = (parsing_anno == 10)[..., None]
        mouth = (parsing_anno == 11)[..., None]
        u_lip = (parsing_anno == 12)[..., None]
        l_lip = (parsing_anno == 13)[..., None]
        neck = (parsing_anno == 14)[..., None]
        neck_l = (parsing_anno == 15)[..., None]

        face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, nose, mouth, u_lip, l_lip))
        mouth_mask = np.logical_or.reduce(( u_lip, l_lip ))
        r_eyes = np.logical_or.reduce((r_brow,r_eye ))
        l_eyes = np.logical_or.reduce((l_brow,l_eye ))
        no_eyes = np.logical_or.reduce((l_eye, r_eye ))

        #mouth
        if 'mouth' in parts:
            mouth_save_path  = os.path.join(base_dir, 'mouth')
            os.makedirs(mouth_save_path, exist_ok=True)
            mouth_uint8 = (mouth_mask.astype(np.uint8) * 255)
            mouth_uint8 = np.squeeze(mouth_uint8)
            cv2.imwrite(os.path.join(mouth_save_path, f'{im_name}.png'), mouth_uint8)

        #eyes
        if 'eyes' in parts:
            r_eyes_uint8 = (r_eyes.astype(np.uint8)  * 255)
            l_eyes_uint8 = (l_eyes.astype(np.uint8)  * 255)

            face_mask = face.astype(np.uint8) * 255
            r_eyes_uint8 = process_eyes(r_eyes_uint8, face_mask)
            l_eyes_uint8 = process_eyes(l_eyes_uint8, face_mask)

            no_eyes = (no_eyes.astype(np.uint8) * 255)
            no_eyes = np.squeeze(no_eyes)
            
            #process eye area
            combined_eyes = cv2.bitwise_or(r_eyes_uint8, l_eyes_uint8)
            combined_eyes = combined_eyes - no_eyes
            combined_eyes = np.squeeze(combined_eyes)

            eyes_save_path = os.path.join(base_dir, 'eyes')
            os.makedirs(eyes_save_path, exist_ok=True)
            cv2.imwrite(os.path.join(eyes_save_path, f'{im_name}.png'), combined_eyes)

        #brows
        if 'brows' in parts:
            brows_save_path = os.path.join(base_dir, 'brows')
            os.makedirs(brows_save_path, exist_ok=True)
            l_brow_uint8 = (l_brow.astype(np.uint8) * 255)
            r_brow_uint8 = (r_brow.astype(np.uint8) * 255)
            combined_brows = cv2.bitwise_or(l_brow_uint8, r_brow_uint8)
            combined_brows = np.squeeze(combined_brows)
            cv2.imwrite(os.path.join(brows_save_path, f'{im_name}.png'), combined_brows)

        # --- NEW: Cheeks Extraction Logic ---
        if 'cheeks' in parts:
            # 1. Data Type Conversion (Boolean -> Uint8)
            # Ensure inputs are single-channel 2D arrays (H, W)
            skin_u8 = (skin.astype(np.uint8) * 255).squeeze()
            nose_u8 = (nose.astype(np.uint8) * 255).squeeze()
            
            # 2. Aggregate Eye Groups (Eyebrow + Eye) for stable BBoxes
            # Note: See "Developer Notes" regarding glasses (Class 6)
            l_eye_group = np.logical_or.reduce((l_brow, l_eye, eye_g)).astype(np.uint8) * 255
            r_eye_group = np.logical_or.reduce((r_brow, r_eye, eye_g)).astype(np.uint8) * 255
            l_eye_group = np.squeeze(l_eye_group)
            r_eye_group = np.squeeze(r_eye_group)

            # 3. Generate Left and Right Cheeks
            # Mapping Logic:
            # Image Left = Subject's Right Cheek -> Reference Right Eye (r_eye_group)
            # Image Right = Subject's Left Cheek -> Reference Left Eye (l_eye_group)
            left_cheek_mask = get_cheek_mask(skin_u8, r_eye_group, nose_u8, face_side='left')
            right_cheek_mask = get_cheek_mask(skin_u8, l_eye_group, nose_u8, face_side='right')

            # 4. Combine and Save
            combined_cheeks = cv2.bitwise_or(left_cheek_mask, right_cheek_mask)
            
            cheeks_save_path = os.path.join(base_dir, 'cheeks')
            os.makedirs(cheeks_save_path, exist_ok=True)
            cv2.imwrite(os.path.join(cheeks_save_path, f'{im_name}.png'), combined_cheeks)

        # --- End of Cheeks Extraction ---


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth', parts='all'):

    os.makedirs(respth, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for image_path in tqdm.tqdm(os.listdir(dspth)):
            if not (image_path.endswith('.jpg') or image_path.endswith('.png') or image_path.endswith('.jpeg')):
                 continue
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)

            img = to_tensor(image).unsqueeze(0).cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(image, parsing, stride=1, parts=parts, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate(respth=args.output_path, dspth=args.data, cp=args.cp, parts=args.parts)
