import os
from ultralytics import YOLO

# === CONFIGURATION ===
image_folder = "/media/wfudronelab/disk/DCIM/100MSDCF"
output_root = "/home/wfudronelab/photos"    # Where each result folder will go
model_path = "yolov8n.pt"            # You can use yolov8s.pt or other models

# === Load YOLO model ===
model = YOLO(model_path)

# === Ensure output folder exists ===
os.makedirs(output_root, exist_ok=True)

# === Process each image ===z
for idx, filename in enumerate(sorted(os.listdir(image_folder))):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        image_path = os.path.join(image_folder, filename)
        
        # Create a unique folder for this image's result
        output_folder = os.path.join(output_root, f"result_{idx+1}")
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing: {image_path} → {output_folder}")
        
        # Run prediction
        model.predict(source=image_path, save=True, project=output_folder, name="")

print("Done processing all images.")
