#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
from threading import Thread
import os
from cancer_detection_model import predict_image  # Import the actual prediction function from your model file

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    img_path = request.args.get('img_path')  # Retrieve image path from the request
    
    if not img_path:
        return jsonify({"error": "Image path not provided"}), 400
    
    if not os.path.exists(img_path):
        return jsonify({"error": "Invalid image path provided"}), 400
    
    try:
        # Call the actual model's prediction function
        result = predict_image(img_path)  # This should return the result from your model
        return jsonify({"result": result, "img_path": img_path})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to run the Flask app
def run_app():
    app.run(port=5000, debug=True, use_reloader=False)  # Use reloader=False to prevent restarting in Jupyter

# Thread wrapper for running the server
thread = Thread(target=run_app)
thread.daemon = True  # Ensures the thread exits when the main program exits
thread.start()

# Graceful shutdown in Jupyter or script
try:
    while True:
        pass  # Keep the main thread alive
except KeyboardInterrupt:
    print("\nStopping server...")
    os._exit(0)  # Forcefully exits all threads (including Flask)
