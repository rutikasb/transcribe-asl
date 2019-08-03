from threading import Thread
import os
import numpy as np
import tensorflow as tf
import base64
import flask
import uuid
import time
import json
import sys
import io
import logging
import cv2
import time
from PIL import Image
from keras.preprocessing.image import img_to_array
from flask_cors import CORS
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model, Model
from PREP_USER_VIDEO import CLIP_SINGLE_VIDEO as clip_single_video


LSTM_MODEL_PATH = './models'
global inceptionModel
global base_inceptionModel
global lstm_model

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
CORS(app)


#Load Tensorflow frozen model
def loadModel():
    global base_inceptionModel
    global lstm_model

    inceptionModel = InceptionV3(weights='imagenet', include_top=True)
    base_inceptionModel = Model(inputs=inceptionModel.input, outputs=inceptionModel.get_layer('avg_pool').output)
    lstm_model = load_model(os.path.join(LSTM_MODEL_PATH, 'video_LSTM.h5'))
    print("Loaded {0}".format(os.path.join(LSTM_MODEL_PATH, 'video_LSTM.h5')))
    global graph
    graph = tf.get_default_graph()



loadModel()

sign_mapping = {0: 'AGAIN',
                1: 'BEAUTIFUL',
                2: 'BOY',
                3: 'CAR',
                4: 'DRINK',
                5: 'FAMILY',
                6: 'HELLO',
                7: 'NAME',
                8: 'WALK'}



def get_cnn_features(frames):
    with graph.as_default():
        featuresList = []
        for frame in frames:
            img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dim = (299, 299)
            img_data = cv2.resize(img_data, dim, interpolation = cv2.INTER_AREA)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            inceptionv3_feature = base_inceptionModel.predict(img_data)
            featuresList.append(inceptionv3_feature)
        stackedFeatures = np.vstack(featuresList)
    return stackedFeatures

def randind(N, n):
    s = [i*N//n for i in range(n)]
    s.append(N)
    nums = []
    for i in range(len(s)-1):
        nums.append(np.random.randint(s[i], s[i+1]))
    return nums


def processVideo(filename, sequenceLength, featureLength):
    print("Processing video file: {0}".format(filename))
    cap = cv2.VideoCapture(filename)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ret, frame = cap.read()
    if ret == False:
        print("Could not open video file: {0}".format(filename))
        return

    randomIndicesToProcess = randind(total_frames, sequenceLength)
    dim = (sequenceLength, featureLength)
    frames = []
    X = np.empty((1, *dim))
    for f in randomIndicesToProcess:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        frames.append(frame)

    X[0,] = get_cnn_features(frames)
    with graph.as_default():
        result = lstm_model.predict(X, verbose=1)
    predicted_sign = sign_mapping[np.argmax(result)]
    predicted_conf = result[0, np.argmax(result)]
    result = []
    result.append({'label': predicted_sign, 'conf': int(100*predicted_conf)})
    print("Predicted sign is {}, with conf {}".format(predicted_sign, 100*predicted_conf))
    cap.release()
    return result


@app.route("/predict/video", methods=["POST"])
def predictVideo():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("video"):
            record_type = flask.request.form.get('record_type', '')
            username = flask.request.form.get('username', '')
            attempted_sign = flask.request.form.get('attempted_sign', '')
            timestamp = int(time.time())
            # read the image in PIL format and prepare it for
            # classification
            videofile = flask.request.files["video"]
            print("Type of videofile: {}".format(type(videofile)))
            filename = "video.avi"
            videofile.save(filename)
            videofile.close()
            clip_single_video(filename, filename)
            sequenceLength = 20
            featureLength = 2048
            predictions = processVideo(filename, sequenceLength, featureLength)
            data["predictions"] = predictions
            # indicate that the request was a success
            data["success"] = True

            print(f'{timestamp},{username},{attempted_sign},{predictions[0]["label"]},{predictions[0]["conf"]}\n')
            with open("logs/api.log", "a") as f:
                f.write(f'{timestamp},{username},{attempted_sign},{predictions[0]["label"]},{predictions[0]["conf"]}\n')

    # return the data dictionary as a JSON response
    return flask.jsonify(data)




# If this is the main thread of execution
# first load the model (already called loadModel(), see above)
# then start the server
if __name__ == "__main__":

    print("* Starting web service...")

    # This is used only when running locally.
    # For google app engine, Gunicorn is used, see entrypoint in app.yaml.
    app.run(host='0.0.0.0', port=8080, debug=True)
    
