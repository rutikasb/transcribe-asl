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


model = InceptionV3(weights='imagenet', include_top=True)
base_model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
_, height, width, channels = base_model.input_shape
print(base_model.input_shape)

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
        height, width, depth = img_data.shape
        sX = int(width/2 - 299/2)
        sY = int(height/2 - 299/2)
        img_data = img_data[sY:sY+299, sX:sX+299, :]
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        inceptionv3_feature = base_model.predict(img_data)
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
    global base_model
    global lstm_model

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
        cv2.imshow('frame',frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    X[0,] = get_cnn_features(frames)
    result = lstm_model.predict(X, verbose=1)
    predicted_sign = sign_mapping[np.argmax(result)]
    predicted_conf = result[0, np.argmax(result)]
    print("Predicted sign is {}, with conf {:0.2f}".format(predicted_sign, predicted_conf))
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN+LSTM model for short video snippets')

    parser.add_argument('--video_path', required=True, help='path to the test video')
    args = vars(parser.parse_args())
    print(args)

    sequenceLength = 6
    featureLength = 2048
    processVideo(args['video_path'], sequenceLength, featureLength)