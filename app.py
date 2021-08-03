import boto3
import urllib.request
from PIL import Image 
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import yaml
import dvc.api 
from io import BytesIO
import os
import requests

app = Flask(__name__)
PEOPLE_FOLDER = os.path.join('static')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

MODEL_PATH = '/home/shruthi/Shruthi_Tasks/image_classifier/mob_model_catdog.h5'
model = load_model(MODEL_PATH)
print(type(model))
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')

def predict_label(img_path,model):
	#Image_Width=128
	#Image_Height=128
	#Image_Size=(Image_Width,Image_Height)
	#Image_Channels=3

	i = image.load_img(img_path, target_size=(128,128))
	i = image.img_to_array(i)
	image_batch = np.expand_dims(i, axis=0)
	processed_image = preprocess_input(image_batch, mode='caffe')
	preds = model.predict(processed_image)
	#i = i.reshape(1, 128,128,3)
	#p = model.predict(i)
	return preds
s3 = boto3.client('s3')
# Generate the URL to get 'key-name' from 'bucket-name'
url = s3.generate_presigned_url(
    ClientMethod='get_object',
    Params={
        'Bucket': 'flask-image-data',
        'Key': 'cat2.jpg'
    }
)
urllib.request.urlretrieve(url, "/home/shruthi/Shruthi_Tasks/s3_flask/static/filename.jpg")
img = Image.open('/home/shruthi/Shruthi_Tasks/s3_flask/static/filename.jpg')

print(type(img))
@app.route('/',methods = ['GET', 'POST'])
def show_index():
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'filename.jpg')
	print(full_filename)

	
	p = predict_label(full_filename,model)
	if np.argmax(p) == 1:
		str_predicted='Dog'
	else:
		str_predicted='Cat'
	return render_template("index.html", prediction = str_predicted, user_image = full_filename)





if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)