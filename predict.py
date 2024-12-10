import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (ensure you specify the correct model path)
model = YOLO('C:/Users/sarfa/Documents/PythonProject/yolov10-object-detection-custom-data/runs/detect/train4/weights/last.pt')
  # Adjust the
# path if necessary

# Load the video file
input_video_path = 'video.mp4'
output_video_path = 'out.mp4'

# Open the video using OpenCV
video_capture = cv2.VideoCapture(input_video_path)

# Check if the video was successfully opened
if not video_capture.isOpened():
    print(f"Error: Unable to open video file {input_video_path}")
    exit()

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out_video = cv2.VideoWriter(output_video_path, fourcc, fps,
                            (frame_width, frame_height))

# Iterate over each frame
frame_count = 0
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        break

    # Apply YOLO object detection
    results = model.predict(frame, verbose=False)

    # Draw bounding boxes on the frame
    for box in results[0].boxes:  # Iterate through detections
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class index
        label = f'{model.names[cls]} {conf:.2f}'

        # Draw bounding box and label if confidence is above threshold
        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out_video.write(frame)

    # Print progress
    frame_count += 1
    print(f'Processed frame {frame_count}/{total_frames}')

# Release resources
video_capture.release()
out_video.release()
cv2.destroyAllWindows()

print(f'Output video saved to {output_video_path}')
