import cv2
import time
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# Initialize Picamera2 for the OV9281 mono camera
picam2 = Picamera2()

# Configure camera stream (1280x800 is native 1MP resolution for OV9281)
config = picam2.create_video_configuration(
    main={"size": (1280, 800), "format": "XRGB8888"}
)
picam2.configure(config)
picam2.start()

def generate_frames():
    while True:
        # Capture raw frame as a NumPy array for OpenCV processing
        frame = picam2.capture_array()

        # Convert to single-channel grayscale (OV9281 is monochrome)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Example OpenCV image processing (e.g., adding timestamp)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(gray, f"OV9281 | {timestamp}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        # Encode frame as JPEG for MJPEG streaming
        ret, buffer = cv2.imencode('.jpg', gray, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Format output as multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Listen on all local interfaces on port 8002
    app.run(host='0.0.0.0', port=8002, threaded=True)