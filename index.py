import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('pneumonia_detection_model2.h5')

# Define the prediction function
def predict_pneumonia(img):
    # Ensure img is a PIL image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Resize the image to the expected input shape
    img = img.resize((150, 150))

    # Convert to NumPy array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Predict using the model
    prediction = model.predict(img_array)
    prediction = prediction.flatten()[0]

    # Mapping prediction to labels
    return {'Normal': float(1 - prediction), 'Pneumonic': float(prediction)}

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title="Pneumonia Detection System",
    description="Upload an image to detect pneumonia. The system will classify the image as Normal or Pneumonic."
)

# Launch the interface
iface.launch(server_name="0.0.0.0", server_port=8080)
