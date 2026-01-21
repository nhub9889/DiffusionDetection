import torch
import cv2
import numpy as np
import argparse
import yaml
import os
from PIL import Image
from torchvision.transforms import v2 as T

from src.core.model import DiffusionDetModel

def get_inference_transform():
    return T.Compose([
        T.Resize((640, 640), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def visualize_prediction(image_path, result, output_path, threshold=0.5):
    img = cv2.imread(image_path)
    h_orig, w_orig, _ = img.shape

    pred_boxes = result['boxes'].cpu().numpy()
    if pred_boxes.ndim == 3: pred_boxes = pred_boxes[0]

    pred_scores = result['scores'].cpu().numpy()
    if pred_scores.ndim == 3: pred_scores = pred_scores[0]

    max_score = np.max(pred_scores)
    print(f"\n[DEBUG] Max Confidence Score detected: {max_score:.4f}")

    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    count = 0
    for i, box in enumerate(pred_boxes):
        class_vector = pred_scores[i]
        class_id = np.argmax(class_vector)
        score = class_vector[class_id]

        if score > threshold:
            count += 1
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_id}: {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, img)
    print(f"Saved result to {output_path} (Found {count} boxes)")


def main(args):
    device = torch.device(args.device)

    # 1. Load Config & Model
    print(f"Loading model from {args.checkpoint}...")
    model = DiffusionDetModel(args.config)

    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Processing {args.image_path}...")
    pil_img = Image.open(args.image_path).convert("RGB")
    transform = get_inference_transform()
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        results = model(img_tensor)
        result = results[0]
    output_filename = "pred_" + os.path.basename(args.image_path)
    visualize_prediction(args.image_path, result, output_filename, threshold=args.threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference DiffusionDet")
    parser.add_argument('--config', default='resnet18.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', default='output/latest.pth', help='Path to trained checkpoint')
    parser.add_argument('--image-path', required=True, help='Path to input image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()
    main(args)