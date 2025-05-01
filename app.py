from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt', source='local')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No image uploaded'

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    results = model(filepath)
    labels = results.pandas().xyxy[0]['name'].tolist()
    unique_labels = list(set(labels))

    return render_template('result.html', labels=unique_labels, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
