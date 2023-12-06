import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
interpreter = tf.lite.Interpreter(model_path="bees-wasps-v2.tflite")
interpreter.allocate_tensors()

# Load and preprocess the image
image_url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"
image = download_image(image_url)  # Assume you have the download_image function from previous code
target_size = (150, 150)  # Adjust based on your model requirements
resized_image = prepare_image(image, target_size)
image_array = np.array(resized_image)
input_array = image_array / 255.0
input_array = np.expand_dims(input_array, axis=0)

# Set the input tensor
input_tensor_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_tensor_index, input_array.astype(np.float32))

# Run the interpreter
interpreter.invoke()

# Get the output tensor
output_tensor_index = interpreter.get_output_details()[0]['index']
model_output = interpreter.get_tensor(output_tensor_index)

# Print the model output
print(f"The output from the model is: {model_output[0][0]:.4f}")
