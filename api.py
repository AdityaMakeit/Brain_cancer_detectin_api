from flask import Flask, request, jsonify
import os
from cancer_detection_model import predict_image  # Must NOT train here, only load & predict

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    img_path = request.args.get('img_path')
    
    if not img_path:
        return jsonify({"error": "Image path not provided"}), 400

    if not os.path.exists(img_path):
        return jsonify({"error": "Invalid image path provided"}), 400

    try:
        result = predict_image(img_path)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the port provided by Render
    app.run(host='0.0.0.0', port=port)
