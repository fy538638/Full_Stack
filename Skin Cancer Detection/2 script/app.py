from flask import Flask, render_template, request,jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io
from PIL import Image

app = Flask(__name__)

# Load the saved model
themodel = tf.saved_model.load('./model/')

# Define the route to handle form submissions
@app.route("/")
def landing():
    return "<h1>welcome" 


def predict(image_bytes):
    # Open the image from binary data and resize it to (299, 299)
    img = Image.open(io.BytesIO(image_bytes)).resize((299, 299))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Normalize the image array
    img_processed = img_array / 255.0
    # Add an extra dimension to the array to match the input shape of the model
    img_processed = np.expand_dims(img_processed, axis=0)

    # Cast the input tensor to a float tensor
    img_processed = tf.cast(img_processed, tf.float32)
    
    # Get the prediction from the model
    result = themodel.signatures['serving_default'](input_3=img_processed)
    return np.argmax(result)

# Define the Flask route for image prediction
@app.route('/predict', methods=['POST'])
def predict_image():
    # Get the binary data of the uploaded image file from the request object
    img_bytes = request.data
    # Get the prediction from the model
    category = {0: 'benign', 1: 'malignant'}
    prediction = category[predict(img_bytes)]
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})



# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)