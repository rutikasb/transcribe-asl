import os
import numpy as np
import glob
import cv2, json
import argparse
from sklearn.utils import shuffle
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.layers import LSTM
from keras.models import load_model, Model
from sequence_data_generator import FramesSeqGenerator, FeaturesSeqGenerator
from data_generator import DataGenerator
from keras import backend as K

from tkinter import filedialog
from tkinter import *


inceptionModel = InceptionV3(weights='imagenet', include_top=True)
base_inceptionModel = Model(inputs=inceptionModel.input, outputs=inceptionModel.get_layer('avg_pool').output)
lstm_model = load_model('video_LSTM.h5')

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
    featuresList = []
    for frame in frames:
        img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dim = (299, 299)
        img_data = cv2.resize(img_data, dim, interpolation = cv2.INTER_AREA)
        # height, width, depth = img_data.shape
        # sX = int(width/2 - 299/2)
        # sY = int(height/2 - 299/2)
        # img_data = img_data[sY:sY+299, sX:sX+299, :]
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
        image = cv2.resize(rgb, (299, 299), interpolation = cv2.INTER_AREA)
        # cv2.imwrite(filename, image)
        ofFrames.append(image)
        prvs = next
        count += 1
    cap.release()
    total_frames = len(ofFrames)
    randomIndicesToProcess = randind(total_frames, sequenceLength)
    dim = (sequenceLength, featureLength)
    frames = []
    X = np.empty((1, *dim))
    for f in randomIndicesToProcess:
        frame = ofFrames[f]
        frames.append(frame)

    # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    # ret, frame = cap.read()
    # if ret == False:
    #     print("Could not open video file: {0}".format(filename))
    #     return

    # randomIndicesToProcess = randind(total_frames, sequenceLength)
    # dim = (sequenceLength, featureLength)
    # frames = []
    # X = np.empty((1, *dim))
    # for f in randomIndicesToProcess:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    #     ret, frame = cap.read()
    #     frames.append(frame)

    X[0,] = get_cnn_features(frames)
    result = lstm_model.predict(X, verbose=1)
    predicted_sign = sign_mapping[np.argmax(result)]
    predicted_conf = result[0, np.argmax(result)]
    print("Predicted sign is {}, with conf {:0.2f}".format(predicted_sign, predicted_conf))

    if(show==True):
        cap = cv2.VideoCapture(filename)
        for f in randomIndicesToProcess:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            cv2.imshow('Video',frame)
            cv2.moveWindow('Video', 900,300)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        text = "{0}, conf {1}".format(predicted_sign, int(predicted_conf*100))
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2, cv2.LINE_AA, )
        cv2.imshow('Video',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()
    



if __name__ == '__main__':
    sequenceLength = 7
    featureLength = 2048

    parser = argparse.ArgumentParser(description='Test CNN+LSTM model for short video snippets')

    parser.add_argument('--video_path', required=False, help='path to the test video')
    args = vars(parser.parse_args())
    if args.get('video_path') is None:
        while True:
            args["video_path"] = filedialog.askopenfilename(initialdir="./Test-Only-Clips", title='Select video file',
                                                    filetypes=[("Video files", "*.avi *.AVI *.mp4 *.MP4")])
            if not args["video_path"]:
                break
            processVideo(args['video_path'], sequenceLength, featureLength, show=True)

    processVideo(args['video_path'], sequenceLength, featureLength, show=True)