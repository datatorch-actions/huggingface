"""Run a model from the Hugging Face Hub on one image and return annotations.

Object detection: each result's box becomes a bbox. Image segmentation:
each result's mask becomes one or more polygons (its outer contours).
Labels are the model's own class names; DataTorch creates any the
project does not have yet.
"""
import os
import sys

import numpy as np
import datatorch
from PIL import Image
from transformers import pipeline

image_path = datatorch.get_input("imagePath")
model = (datatorch.get_input("model") or "").strip()
task = (datatorch.get_input("task") or "object-detection").strip()
threshold = float(datatorch.get_input("threshold") or 0.5)

if not model:
    print("No model id given.")
    sys.exit(1)
if task not in ("object-detection", "image-segmentation"):
    print(f"Unsupported task '{task}': use object-detection or image-segmentation.")
    sys.exit(1)

image = Image.open(image_path).convert("RGB")
print(f"Loading {model} for {task}")
kwargs = {}
token = os.environ.get("HF_TOKEN")
if token:
    kwargs["token"] = token
runner = pipeline(task, model=model, **kwargs)

annotations = []
if task == "object-detection":
    for result in runner(image, threshold=threshold):
        box = result["box"]
        x, y = box["xmin"], box["ymin"]
        width, height = box["xmax"] - x, box["ymax"] - y
        if width <= 0 or height <= 0:
            continue
        annotations.append(
            {
                "label": str(result["label"]),
                "bbox": [x, y, width, height],
                "score": float(result["score"]),
            }
        )
else:
    import cv2

    for result in runner(image):
        score = result.get("score")
        if score is not None and float(score) < threshold:
            continue
        mask = np.array(result["mask"].convert("L")) > 127
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rings = []
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) < 4:
                continue
            rings.append([[int(p[0][0]), int(p[0][1])] for p in contour])
        if not rings:
            continue
        annotation = {"label": str(result["label"]), "polygons": rings}
        if score is not None:
            annotation["score"] = float(score)
        annotations.append(annotation)

print(f"Found {len(annotations)} annotation(s)")
datatorch.set_output("annotations", annotations)
