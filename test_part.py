import tensorflow as tf
import numpy as np
import os

files=[]
for f in os.scandir('/Users/aronovmihail/Documents/'): # path to the directory with images you want to test
    if f.is_file() and f.path.split('.')[-1].lower() == 'png':
        files.append(f.path)
model = tf.keras.models.load_model('/Users/aronovmihail/Documents/model') # path to yourn model
img_height = 593 # change this to be like in learn_part.py
img_width = 904  # change this to be like in learn_part.py
class_names = ['QSO', 'galaxies', 'stars'] # change this to be like the learn_part.py prints out on some of the early lines (like ['QSO', 'galaxies', ...])
try:
    for each in files:
        img = tf.keras.utils.load_img(each, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(each + ": this image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
except FileNotFoundError:
    print('Incorrect file adress. Please try again.')