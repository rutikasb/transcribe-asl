# curl -X POST -F 'username=chandan' -F 'attempted_sign=again' -F "video=@/home/chandangope/py/temp/transcribe-asl/raw_data/train/AGAIN/1563315188711.avi" https://us-central1-w210-capstone-242920.cloudfunctions.net/video1

from google.cloud import storage
import time
import tensorflow as tf
import numpy as np
import cv2
import argparse
from keras.layers import LSTM
from keras.applications import MobileNet
from keras.models import load_model, Model
from keras.applications.mobilenet import preprocess_input

lstm_model = None
model=MobileNet(weights='imagenet',include_top=True)
base_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)

sign_mapping = {0: 'AGAIN',
                1: 'BEAUTIFUL',
                2: 'BEST',
                3: 'BOY',
                4: 'CAR',
                5: 'DRINK',
                6: 'FAMILY',
                7: 'HELLO',
                8: 'HOT',
                9: 'NAME',
                10: 'SAY',
                11: 'TIRED',
                12: 'WALK'}

def get_cnn_features(frames):
    global base_model
    featuresList = []
    for frame in frames:
        img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dim = (224, 224)
        img_data = cv2.resize(img_data, dim, interpolation = cv2.INTER_AREA)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        mobilenet_feature = base_model.predict(img_data)
        featuresList.append(mobilenet_feature)
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
    global lstm_model

    print("Processing video file: {0}".format(filename))
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, image = cap.read()
    if success == False:
        print("Could not open video file: {0}".format(filename))
        return

    prvs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image)
    hsv[...,1] = 255
    count = 0
    ofFrames = []

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, fps, (299,299))
    while success:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        success, frame2 = cap.read()
        if not success:
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 9, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        image = cv2.resize(rgb, (224, 224), interpolation = cv2.INTER_AREA)
        ofFrames.append(image)
        out.write(image)
        prvs = next
        count += 1
    cap.release()
    out.release()
    total_frames = len(ofFrames)
    randomIndicesToProcess = randind(total_frames, sequenceLength)
    dim = (sequenceLength, featureLength)
    frames = []
    X = np.empty((1, *dim))
    for f in randomIndicesToProcess:
        frame = ofFrames[f]
        frames.append(frame)

    X[0,] = get_cnn_features(frames)
    result = lstm_model.predict(X, verbose=1)
    predicted_sign = sign_mapping[np.argmax(result)]
    predicted_conf = result[0, np.argmax(result)]
    print("Predicted sign is {}, with conf {:0.2f}".format(predicted_sign, predicted_conf))
    cap.release()
    
    
def processPostVideo(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    global lstm_model
    
    request_json = request.get_json()
    print(request.headers)
    print('files:')
    for f in request.files:
        print(f)
    if request.files.get("video"):
        record_type = request.form.get('record_type', '')
        username = request.form.get('username', '')
        attempted_sign = request.form.get('attempted_sign', '')
        print("attempted_sign="+attempted_sign)
        videofile = request.files["video"]
        
        gcs = storage.Client()
        bucket_name = "w210-asl-attempted-signs-" + attempted_sign.lower()
        print("bucket_name="+bucket_name)
        buckets = gcs.list_buckets()
        buckets = [bucket.name for bucket in buckets]
        print(buckets)
        if(bucket_name not in buckets):
            bucket = gcs.create_bucket(bucket_name)
            print('Bucket {} created.'.format(bucket.name))
        else:
            bucket = gcs.get_bucket(bucket_name)
            print('Bucket {} already exists.'.format(bucket_name))
        
        blob = bucket.blob(videofile.filename)
        blob.upload_from_string(videofile.read(), content_type=videofile.content_type)
        print('File {} uploaded to {}'.format(videofile.filename, bucket_name))
        
        if lstm_model is None:
            print("Loading LSTM model from {0}".format("/tmp/video_LSTM.h5"))
            download_blob(source_blob_name="video_LSTM.h5",
                          destination_file_name="/tmp/video_LSTM.h5", bucket_name="w210-asl-model")
            lstm_model = load_model("/tmp/video_LSTM.h5")
        print("LSTM model loaded from {0}".format("/tmp/video_LSTM.h5"))
        print("Downloading videoblob {0}".format(videofile.filename))
        download_blob(source_blob_name=videofile.filename,
                          destination_file_name="/tmp/video.mp4", bucket_name=bucket_name)
        print("Videoblob downloaded from bucket {0}".format(bucket_name))
        sequenceLength = 7
        featureLength = 1024
        processVideo("/tmp/video.mp4", sequenceLength, featureLength, show=False)
        
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return f'Hello World!'

    
    
def download_blob(source_blob_name, destination_file_name, bucket_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project='iv-automl-test')
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))