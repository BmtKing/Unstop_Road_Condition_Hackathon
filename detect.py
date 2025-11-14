from ultralytics import YOLO
import cv2
import pandas as pd
import os

MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

print(f"Loaded custom model from {MODEL_PATH}")

TEST_IMAGES_DIR = 'testimages/'

image_files = os.listdir(TEST_IMAGES_DIR)

all_detections = []

print(f"Running detection on {len(image_files)} images...")

for image_file in image_files:
    image_path = os.path.join(TEST_IMAGES_DIR, image_file)
    
    results = model(image_path)
    
    img = cv2.imread(image_path)
    
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            
            all_detections.append({
                'image': image_file,
                'class': cls_name,
                'confidence': conf
            })
            
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(img, f'{cls_name} {conf:.2f}', (b[0], b[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if image_files.index(image_file) :
        print(f"--- Displaying result for {image_file} ---")
        cv2.imshow(f'Result for {image_file}', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("\n--- Detection Complete ---")

if all_detections:
    df = pd.DataFrame(all_detections)

    print("\n--- Analysis Report ---")
    
    print(f"Total items of damage found: {len(df)}")

    damage_by_type = df['class'].value_counts()
    print("\nDamage Counts by Type:")
    print(damage_by_type)

    avg_confidence = df.groupby('class')['confidence'].mean()
    print("\nAverage Confidence by Type:")
    print(avg_confidence)
else:
    print("No detections found in the sample images.")