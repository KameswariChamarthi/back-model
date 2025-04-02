from flask import Flask, request, jsonify, send_from_directory
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
from database import db, ScanResult
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])

# Set up logging
logging.basicConfig(level=logging.INFO)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepfake_results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create Uploads Folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load the TFLite Model at the START
try:
    model_path = r"D:\New folder\deepfake_detection\deepfake_detector.tflite"

    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    print("Expected Input Shape:", input_details[0]['shape'])  # Debugging

    logging.info("✅ Model Loaded Successfully!")
except Exception as e:
    logging.error(f"🚨 ERROR: Failed to load model: {e}")
    model = None  # Prevents usage if loading fails

@app.route('/')
def home():
    return "API is working!"

# ✅ API Status Check
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"message": "API is running!"})

# ✅ Store Deepfake Scan Results
@app.route('/store_result', methods=['POST'])
def store_result():
    data = request.json
    filename = data.get('filename')
    prediction = data.get('prediction')
    confidence = data.get('confidence')

    if not filename or not prediction or confidence is None:
        return jsonify({"error": "Missing data"}), 400

    new_result = ScanResult(filename=filename, prediction=prediction, confidence=confidence)
    db.session.add(new_result)
    db.session.commit()

    return jsonify({"message": "Result stored successfully!"}), 201

# ✅ Upload Image
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    return jsonify({"message": "Upload successful", "image_url": f"/images/{image.filename}"}), 201

# ✅ Serve Uploaded Images
@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ Deepfake Detection Route
@app.route('/detect-deepfake', methods=['POST'])
@app.route('/predict', methods=['POST'])  # Alias for prediction
def detect_deepfake():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    try:
        image = request.files.get('image')

        if not image:
            return jsonify({"error": "No image file provided"}), 400

        allowed_extensions = {"png", "jpg", "jpeg"}
        if not image.filename.lower().endswith(tuple(allowed_extensions)):
            return jsonify({"error": "Invalid image file type"}), 400

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # ✅ Fix Image Preprocessing (Ensure correct input size)
        img = Image.open(image_path).convert("RGB").resize((299, 299))  # Ensure correct size
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)

        # Debugging: Print final image shape
        print("Final Image Shape:", img_array.shape)

        # ✅ Run Model Inference
        model.set_tensor(input_details[0]["index"], img_array)
        model.invoke()
        output_data = model.get_tensor(output_details[0]["index"])

        # Debugging: Print raw model output
        print("Raw Model Output:", output_data)

        # ✅ Fix Confidence Value (Normalize to 0-1)
        confidence = (float(output_data[0][0]) + 1) / 2  # Convert to range [0,1]

        prediction = "Deepfake" if confidence > 0.5 else "Real"

        return jsonify({
            "is_deepfake": prediction,
            "confidence": confidence,
            "processed_image": image.filename
        })

    except Exception as e:
        logging.error(f"🚨 ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    

# ✅ Run Flask App
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Ensure database tables are created
    app.run(debug=True)
