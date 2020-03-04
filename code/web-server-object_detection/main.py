"""
Make sure the file 'models/video_LSTM.h5' is present
Make sure there is a folder 'logs'

Sample curl command:
curl -X POST -F 'username=username' -F 'record_type=record_type' -F 'attempted_sign=NAME' -F "video=@/home/chandangope/py/temp/transcribe-asl/Test-Only-Clips/newClips/NAME/WIN_20190811_13_49_33_Pro.mp4" localhost:8080/predict/video
"""


from threading import Thread
import os
from shutil import copyfile
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
from google.cloud import storage


LSTM_MODEL_PATH = './models'
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



# loadModel()

sign_mapping = {0: 'AGAIN',
                1: 'BEAUTIFUL',
                2: 'BOY',
                3: 'CAR',
                4: 'DRINK',
                5: 'FAMILY',
                6: 'HELLO',
                7: 'NAME',
                8: 'WALK'}

# if not os.path.exists('user_videos'):
#     os.mkdir('user_videos')
# for key, value in sign_mapping.items():
#     if not os.path.exists(f'user_videos/{value}'):
#         os.mkdir(f'user_videos/{value}')

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


def processVideo(filename, sequenceLength, featureLength, show=False):
    retValue = []
    print("Processing video file: {0}".format(filename))
    cap = cv2.VideoCapture(filename)
    success, image = cap.read()
    if success == False:
        print("Could not open video file: {0}".format(filename))
        return

    prvs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image)
    hsv[...,1] = 255
    count = 0
    ofFrames = []
    while success:
        success, frame2 = cap.read()
        if not success:
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 9, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        image = cv2.resize(rgb, (299, 299), interpolation = cv2.INTER_AREA)
        ofFrames.append(image)
        prvs = next
        count += 1
    cap.release()
    total_frames = len(ofFrames)
    if(total_frames > sequenceLength):
        randomIndicesToProcess = randind(total_frames, sequenceLength)
        dim = (sequenceLength, featureLength)
        frames = []
        X = np.empty((1, *dim))
        for f in randomIndicesToProcess:
            frame = ofFrames[f]
            frames.append(frame)

        X[0,] = get_cnn_features(frames)
        with graph.as_default():
            result = lstm_model.predict(X, verbose=1)
        predicted_sign = sign_mapping[np.argmax(result)]
        predicted_conf = result[0, np.argmax(result)]
        
        retValue.append({'label': predicted_sign, 'conf': int(100*predicted_conf)})
        print("Predicted sign is {}, with conf {}".format(predicted_sign, 100*predicted_conf))
    else:
        print("total_frames is less than sequenceLength, cannot process")

    return retValue


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
            videofile = flask.request.files["video"]
            filename = "video.avi"

            # Create a Cloud Storage client.
            gcs = storage.Client()

            bucket_name = "w210-asl-attempted-signs-" + attempted_sign.lower()
            buckets = gcs.list_buckets()
            buckets = [bucket.name for bucket in buckets]
            print(buckets)
            if(bucket_name not in buckets):
                bucket = gcs.create_bucket(bucket_name)
                print('Bucket {} created.'.format(bucket.name))
            else:
                bucket = gcs.get_bucket(bucket_name)
                print('Bucket {} already exists.'.format(bucket_name))

            # Create a new blob and upload the file's content.
            print(videofile.filename)
            blob = bucket.blob(videofile.filename)
            blob.upload_from_string(videofile.read(), content_type=videofile.content_type)



            # videofile.save(filename)
            # loggedVideoFile = os.path.join('user_videos', attempted_sign.upper(), str(timestamp) + '.avi')
            # copyfile(filename, loggedVideoFile)
            # videofile.close()
            # clip_single_video(VIDEO_FILE=filename, OUTPUT_FILE=filename, ADDITIONAL_LEADING_FRAMES_TO_CLIP=8, ADDITIONAL_TRAILING_FRAMES_TO_CLIP=10)
            # sequenceLength = 7
            # featureLength = 2048
            # predictions = processVideo(filename, sequenceLength, featureLength)
            # data["predictions"] = predictions
            # # indicate that the request was a success
            # data["success"] = True

            # print(f'{timestamp},{username},{attempted_sign},{predictions[0]["label"]},{predictions[0]["conf"]}\n')
            # logfilename = 'logs/api.log'
            # if os.path.exists(logfilename):
            #     append_write = 'a' # append if already exists
            # else:
            #     append_write = 'w' # make a new file if not
            # with open(logfilename, append_write) as f:
            #     f.write(f'{timestamp},{username},{attempted_sign},{predictions[0]["label"]},{predictions[0]["conf"]}\n')

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
    
