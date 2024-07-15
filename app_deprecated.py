from flask import Flask, render_template, Response
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from deep_emotion import DeepEmotion
from PIL import Image
import numpy as np

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

net = DeepEmotion()
net.load_state_dict(torch.load('./deep_emotion-1-10-0.001.pt'))
net.to(device)
net.eval()

classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

def load_image(image):
    image = Image.fromarray(image).convert('L')
    image = transformation(image).float()
    image = image.unsqueeze(0)
    return image.to(device)

def generate_frames():
    face_cascade = cv2.CascadeClassifier('cascade_model/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                roi = frame[y:y + h, x:x + w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (48, 48))
                # roi_resized = np.expand_dims(roi_resized, axis=-1)
                roi_resized = np.array(roi_resized, dtype=np.uint8)
                processed_roi = load_image(roi_resized)
                output = net(processed_roi)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, 1)
                prediction = classes[predicted_class.item()]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
