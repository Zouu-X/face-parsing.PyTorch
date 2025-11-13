import sys
import os
import os.path as osp
from typing import Tuple

import argparse
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms

# Try importing model locally; fall back to repo subdir layout if needed.
try:
    from model import BiSeNet  # type: ignore
except Exception:
    sys.path.append(osp.join(osp.dirname(__file__), 'face-parsing.PyTorch'))
    from model import BiSeNet  # type: ignore


def build_net(cp: str, device: torch.device) -> torch.nn.Module:
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    state = torch.load(cp, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def infer_parsing(net: torch.nn.Module, pil_img: Image.Image, device: torch.device) -> np.ndarray:
    """Run model and return parsing map (H x W) as int np array at 512x512.

    The model is trained/evaluated at 512, so we upsample input to 512 first.
    """
    to_tensor = get_transform()
    img_512 = pil_img.resize((512, 512), Image.BILINEAR)
    inp = to_tensor(img_512).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(inp)[0]  # logits: (1, C, H, W)
        parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
    return parsing  # shape: (512, 512)


def make_face_mask(parsing: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Build binary face mask from parsing labels and resize to out_size (W, H).

    Face mask = skin + brows + eyes + eye_g + nose + mouth + lips
    Labels (CelebAMask-HQ/BiSeNet):
      1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g,
      10 nose, 11 mouth, 12 u_lip, 13 l_lip
    """
    face_bool = (
        (parsing == 1)  # skin
        | (parsing == 2) | (parsing == 3)  # brows
        | (parsing == 4) | (parsing == 5)  # eyes
        | (parsing == 6)  # eye_g
        | (parsing == 10)  # nose
        | (parsing == 11)  # mouth
        | (parsing == 12) | (parsing == 13)  # lips
    )

    face_uint8 = (face_bool.astype(np.uint8) * 255)
    # Resize to requested output size using nearest-neighbor to preserve labels
    out_w, out_h = out_size
    mask_resized = cv2.resize(face_uint8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized


def main():
    parser = argparse.ArgumentParser(description='Extract face masks from parsing maps')
    parser.add_argument('--data', '--input', dest='inp_dir', required=True,
                        help='Path to input images directory (256x256 PNGs)')
    parser.add_argument('--output_path', '--output', dest='out_dir', required=True,
                        help='Directory to write face masks')
    parser.add_argument('--model', '--checkpoint', '--cp', dest='cp', required=True,
                        help='Path to pretrained model .pth file')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_net(args.cp, device)

    # Process only image-looking files to be safe
    names = sorted([n for n in os.listdir(args.inp_dir)
                    if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])

    for name in names:
        in_path = osp.join(args.inp_dir, name)
        try:
            img = Image.open(in_path).convert('RGB')
        except Exception:
            # Skip unreadable files
            continue

        # Run parsing at 512, then build mask and resize to 256x256 (or original size if 256x256 is different)
        parsing = infer_parsing(net, img, device)

        # Determine desired output size: if input is 256x256 as specified, use that; otherwise, use input size
        out_w, out_h = img.size
        mask = make_face_mask(parsing, (out_w, out_h))

        base, _ = osp.splitext(name)
        out_name = base + '.png'
        out_path = osp.join(args.out_dir, out_name)
        cv2.imwrite(out_path, mask)


if __name__ == '__main__':
    main()

