import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from keras._tf_keras.keras.preprocessing import image
from keras.api.models import model_from_json
import numpy as np
import os

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.weights.h5")
print("Loaded model from disk")

def classify(img_file):
    """
    This function takes an image file as argument and classifies it according to the model loaded from disk.

    Parameters:
    img_file (str): The name of the image file to classify.

    Returns:
    None

    Notes:
    The model was trained using the Keras API.
    """
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (256, 256), color_mode='grayscale')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1

    classes=["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
    result = classes[max_prob - 1]

    expected_result = os.path.basename(img_name).split('_')[0]
    print("Image name: ",os.path.basename(img_name))
    print("Predicted Array: ",arr)
    print("Expected Result: ",expected_result,", Predicted Result: ",result)


path = 'HandGestureDataset/test'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.png' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')
