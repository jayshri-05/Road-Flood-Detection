
from ultralytics import YOLO
from tkinter import Tk, filedialog
import cv2
import numpy as np

# Load your trained model
model = YOLO("runs/road_flood_model/weights/best.pt")

# Upload image
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Flood Image",
    filetypes=[("Images", "*.jpg *.jpeg *.png")]
)
root.destroy()

if not image_path:
    print("No image selected.")
    exit()

# Run YOLO flood detection
results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    verbose=False
)

result = results[0]
image = result.orig_img.copy()

height, width = image.shape[:2]

# Create combined water mask
water_mask = np.zeros(
    (height, width), dtype=np.uint8
)

if result.masks is not None:

    for polygon in result.masks.xy:
        points = np.asarray(
            polygon, dtype=np.int32
        )

        if len(points) >= 3:
            cv2.fillPoly(
                water_mask,
                [points],
                255
            )

# Calculate water percentage
water_pixels = cv2.countNonZero(water_mask)
total_pixels = height * width

flood_percentage = (
    water_pixels / total_pixels
) * 100

# Display result with YOLO segmentation
output = result.plot()

# Demo risk levels
if flood_percentage > 80:
    status = "CRITICAL - DEMO"
    advice = "EVACUATE IMMEDIATELY - DEMO ONLY"
    color = (0, 0, 255)

elif flood_percentage > 50:
    status = "HIGH RISK - DEMO"
    advice = "TAKE PRECAUTIONARY MEASURES"
    color = (0, 140, 255)

elif flood_percentage > 30:
    status = "CAUTION - DEMO"
    advice = "MONITOR ROAD CONDITIONS"
    color = (0, 255, 255)

else:
    status = "LOW WATER COVERAGE"
    advice = "CONTINUE MONITORING"
    color = (0, 255, 0)

# Add dark panel at top
cv2.rectangle(
    output,
    (0, 0),
    (width, 115),
    (20, 25, 35),
    -1
)

cv2.putText(
    output,
    f"Water Coverage: {flood_percentage:.2f}%",
    (15, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

cv2.putText(
    output,
    f"Status: {status}",
    (15, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    color,
    2
)

cv2.putText(
    output,
    advice,
    (15, 102),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.52,
    color,
    2
)

print(f"Detected water coverage: {flood_percentage:.2f}%")
print(f"Status: {status}")
print(f"Advice: {advice}")

# Resize only for convenient display
display = output.copy()

if width > 1000 or height > 700:
    scale = min(1000 / width, 700 / height)
    display = cv2.resize(
        output,
        None,
        fx=scale,
        fy=scale
    )

cv2.imshow("Road Flood Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
