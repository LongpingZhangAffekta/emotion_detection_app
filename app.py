import argparse
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from deep_emotion import DeepEmotion
from data_loaders import PlainDataset
from flask import Flask, render_template, Response, request, jsonify
import os
import io

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = 'deep_emotion-1-10-0.001.pt'  # Update this path
net = DeepEmotion()  # Replace with your actual model class
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()

classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# Transform (normalize) the image
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48
    image = transformation(image).float()
    image = image.unsqueeze(0)
    return image.to(device)

def get_emotion(image_bytes):
    img = transform_image(image_bytes)
    with torch.no_grad():
        outputs = net(img)
        _, predicted_class = torch.max(outputs, 1)
        prediction = classes[predicted_class.item()]
    return prediction

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_processed_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    roi = cv2.resize(roi, (48, 48))
                    img = transform_image(cv2.imencode('.jpg', roi)[1].tobytes())
                    with torch.no_grad():
                        outputs = net(img)
                        _, predicted_class = torch.max(outputs, 1)
                        prediction = classes[predicted_class.item()]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_feed')
def processed_feed():
    return Response(gen_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image_bytes = file.read()
        prediction = get_emotion(image_bytes)
        return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)