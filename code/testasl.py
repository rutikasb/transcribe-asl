import numpy as np
import cv2
import argparse
from keras.layers import LSTM
from keras.applications import MobileNet
from keras.models import load_model, Model
from keras.applications.mobilenet import preprocess_input

from tkinter import filedialog
from tkinter import *


model=MobileNet(weights='imagenet',include_top=True)
base_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)
lstm_model = load_model('video_LSTM.h5')

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
        # cv2.imwrite(filename, image)
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
    return "Predicted sign is {}, with conf {:0.2f}".format(predicted_sign, predicted_conf)
    



if __name__ == '__main__':
    sequenceLength = 7
    featureLength = 1024

    parser = argparse.ArgumentParser(description='Test CNN+LSTM model for short video snippets')

    parser.add_argument('--video_path', required=False, help='path to the test video')
    args = vars(parser.parse_args())
    if args.get('video_path') is None:
        while True:
            args["video_path"] = filedialog.askopenfilename(initialdir="../data/test", title='Select video file',
                                                    filetypes=[("Video files", "*.avi *.AVI *.mp4 *.MP4")])
            if not args["video_path"]:
                break
            result = processVideo(args['video_path'], sequenceLength, featureLength, show=True)
            print(result)

    result = processVideo(args['video_path'], sequenceLength, featureLength, show=True)