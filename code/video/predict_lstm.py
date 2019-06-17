import os
import json
import cv2
import argparse
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model

def get_frames(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    success, image = cap.read()
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
        frames.append(image)
        success, image = cap.read()
    cap.release()
    return frames

def predict(filename, model_path, labels_path):
    frames = get_frames(filename)
    x = np.array(frames)
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    model = load_model(model_path)

    x_features = base_model.predict(x)
    x_features = x_features.reshape(x_features.shape[0], x_features.shape[1] * x_features.shape[2], x_features.shape[3])
    predictions = model.predict_classes(x_features)

    with open(labels_path, 'r') as f:
        labels = json.load(f)
    rev_labels = {v:k for k, v in labels.items()}

    class_counts = np.bincount(predictions)
    top_5 = class_counts.argsort()[-5:][::-1]
    results = []
    for index in top_5:
        results.append(rev_labels[index])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a video snippet, predict the class')

    parser.add_argument('--video-file',  required=True, help='path to the video file')
    parser.add_argument('--model', required=True, help='path to saved model')
    parser.add_argument('--labels', required=False, default='labels.txt', help='path to stored labels file')
    args = vars(parser.parse_args())
    if args.get('video_file') is None or args.get('model') is None:
        print('Please provide path to both video file and model on which to predict')
        exit()

    results = predict(args['video_file'], args['model'], args['labels'])
    print(results)
