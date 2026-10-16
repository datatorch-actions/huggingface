"""Run a model from the Hugging Face Hub on one image.

Object detection: each result's box becomes a bbox. Image segmentation:
each result's mask becomes one or more polygons (its outer contours).
Both return `annotations`; labels are the model's own class names, and
DataTorch creates any the project does not have yet. Image
classification returns `classification`: the top label with its score
and the runners-up, kept on the run (the file's metadata is yours).
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
TASKS = ("object-detection", "image-segmentation", "image-classification")
if task not in TASKS:
    print(f"Unsupported task '{task}': use one of {', '.join(TASKS)}.")
    sys.exit(1)

image = Image.open(image_path).convert("RGB")
kwargs = {}
token = os.environ.get("HF_TOKEN")
if token:
    kwargs["token"] = token

# The model's architecture says what it can do; a wrong task fails deep
# inside Transformers with an unreadable error, so check first.
TASK_OF_HEAD = (
    ("ForObjectDetection", "object-detection"),
    ("ForZeroShotObjectDetection", "object-detection"),
    ("ForImageClassification", "image-classification"),
    ("ForUniversalSegmentation", "image-segmentation"),
    ("ForInstanceSegmentation", "image-segmentation"),
    ("ForSemanticSegmentation", "image-segmentation"),
    ("ForImageSegmentation", "image-segmentation"),
    ("ForSegmentation", "image-segmentation"),
)
try:
    from transformers import AutoConfig

    architectures = list(AutoConfig.from_pretrained(model, **kwargs).architectures or [])
except Exception as ex:  # a private or missing model
    print(f"Could not read the model {model} from the Hub: {ex}")
    sys.exit(1)
supported = {t for arch in architectures for head, t in TASK_OF_HEAD if arch.endswith(head)}
if architectures and supported and task not in supported:
    print(
        f"{model} is built for {' or '.join(sorted(supported))} "
        f"({', '.join(architectures)}), not {task}. Set the task input to "
        f"{sorted(supported)[0]}, or pick a model made for {task}."
    )
    sys.exit(1)

print(f"Loading {model} for {task}")
runner = pipeline(task, model=model, **kwargs)

if task == "image-classification":
    ranked = sorted(runner(image, top_k=5), key=lambda r: -float(r["score"]))
    top = [{"label": str(r["label"]), "score": float(r["score"])} for r in ranked]
    print(f"Top: {top[0]['label']} ({top[0]['score']:.2f})")
    datatorch.set_output(
        "classification", {"label": top[0]["label"], "score": top[0]["score"], "top": top}
    )
    sys.exit(0)

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
