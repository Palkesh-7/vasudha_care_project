
from flask import Flask,request, render_template, Response,Markup
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import datetime
import os
import tensorflow as tf
import base64
import PIL.Image as Image
from utils.disease import disease_dic 

global model




app = Flask(__name__)

class_names = ['Potato___Early_blight',
              'Potato___Late_blight',
              'Potato___healthy']



model = load_model('potatoes.h5')
model.make_predict_function()


crop_recommendation_model_path = 'model/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


try:
    os.mkdir('static/shot')
except OSError as error:
    pass

try:
    os.mkdir('static/upload')
except OSError as error:
    pass


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def predict_label(image):
    

    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))


    image = image/255 

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}


@app.route("/")
def home():
    title = 'Vasudha - Home'
    return render_template('index.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    return render_template("disease_detaction.html")


@app.route('/jsTakePic',methods=['POST','GET'])
def jsTakePic():
    return render_template('jsTakePic.html')

  
@app.route("/shot",methods = ['GET', 'POST'])
def shot():
    title = 'Vasudha - Disease Detection'
    if request.method == 'POST':
        
        data = request.form['image']
        print(data)

                 
        data = data.split(',')[1]

        print(data)
       
  
        # Using bytes(str, enc)
        # convert string to byte 
        res = bytes(data, 'utf-8')
        
        # print result
        now = datetime.datetime.now()
        img_path2 = os.path.sep.join(['static/shot', "shot_{}.jpg".format(str(now).replace(":",''))])	

       

        with open(img_path2, "wb") as fh:
            fh.write(base64.decodebytes(res))

        
        result2 = predict_label(img_path2)
        treatment = Markup(str(disease_dic[result2["class"]]))

        
        

        if (result2["confidence"]>=70):
            return render_template("output.html", prediction=result2["class"],confidence = result2["confidence"],img_path = img_path2,treatment = treatment,title=title)
        else:
            return render_template("output.html",msg = "No Match Found")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    title = 'Vasudha - Disease Detection'
    if request.method == 'POST':
        img1 = request.files['my_image']
        img_path1 = "static/upload" + img1.filename	
        img1.save(img_path1)
        result1 = predict_label(img_path1)
        treatment = Markup(str(disease_dic[result1["class"]]))

        if (result1["confidence"]>=0.6):
            return render_template("output.html", prediction=result1["class"],confidence = result1["confidence"],img_path = img_path1,treatment = treatment,title=title)

        else:
            return render_template("output.html",msg = "No Match Found")




@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Vasudha - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Vasudha - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Vasudha - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Vasudha - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


if __name__ =='__main__':
    app.run(debug = True)