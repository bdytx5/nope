from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
mtcnn = MTCNN()

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    image_url = request.json['image_url']
    
    # Load image from url
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Detect faces in the image
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        return jsonify({"message": f"Detected {len(boxes)} faces"}), 200
    else:
        return jsonify({"message": "No faces detected"}), 200

if __name__ == '__main__':
    app.run(debug=True)

