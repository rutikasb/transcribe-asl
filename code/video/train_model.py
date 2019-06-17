import os
import numpy as np
import cv2, json
import argparse
from sklearn.utils import shuffle
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.applications.inception_v3 import InceptionV3
from keras.layers import LSTM
from keras.models import load_model

BATCH_SIZE = 32
SEED = 42

# Based on https://keras.io/preprocessing/image/
def preprocess_data(data_path):
    datagen = ImageDataGenerator(rescale=1./ 255)
    train_generator = datagen.flow_from_directory(
            f'{data_path}/train',
            target_size=(224, 224),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False)

    validation_generator = datagen.flow_from_directory(
            f'{data_path}/test',
            target_size=(224, 224),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False)
    return train_generator, validation_generator

def extract_cnn_features(generator, base_model, generator_type, reload_features=True):
    data = None
    labels = None
    if reload_features == True or not os.path.exists(f'{generator_type}_cnn_features.npy'):
        for i in range(len(generator)):
            x = generator[i][0]
            y = generator[i][1]

            if data is None:
                data = base_model.predict(x)
                labels = y
            else:
                data = np.append(data, base_model.predict(x), axis = 0)
                labels = np.append(labels, y, axis = 0)
        data, labels = shuffle(data, labels)
        np.save(open(f'{generator_type}_cnn_features.npy', 'wb'), data)
        np.save(open(f'{generator_type}_cnn_labels.npy', 'wb'), labels)
    else:
        print("Returning pre-saved CNN features")
        # return saved features
        data = np.load(f'{generator_type}_cnn_features.npy')
        labels = np.load(f'{generator_type}_cnn_labels.npy')

    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], data.shape[3])
    return data, labels

def train_lstm(train_data, train_labels, validation_data, validation_labels, classes, epochs):
    if epochs:
        epochs = int(epochs)
    model = Sequential()
    model.add(LSTM(256, dropout=0.2, input_shape=(train_data.shape[1], train_data.shape[2])))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.5, seed=SEED))
    model.add(Dense(classes, activation='softmax'))
    sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
    model.fit(train_data,
            train_labels,
            validation_data=(validation_data,validation_labels),
            batch_size=BATCH_SIZE,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=True,
            verbose=1)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN+LSTM model for short video snippets')

    parser.add_argument('--data-path', required=True, help='path to the train and test directories')
    parser.add_argument('--lstm-epochs', default=5, required=False, help='number of LSTM epochs')
    parser.add_argument('--reload-cnn-features', default=True, required=False, help='re-extract CNN features from frames')
    args = vars(parser.parse_args())
    print(args)
    if args.get('data_path') is None:
        print('Please provide path to video frames')
        exit()

    train_data_generator, validation_data_generator = preprocess_data(args['data_path'])
    with open('labels.txt', 'w') as f:
        json.dump(train_data_generator.class_indices, f)

    # load inception model
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    print('Base model (Inception V3) summary')
    print(base_model.summary)

    train_data, train_labels = extract_cnn_features(train_data_generator, base_model, 'train', args['reload_cnn_features'])
    validation_data, validation_labels = extract_cnn_features(validation_data_generator, base_model, 'validation', args['reload_cnn_features'])
    model = train_lstm(train_data, train_labels, validation_data, validation_labels, train_data_generator.num_classes, args['lstm_epochs'])
    model.save("final_model.h5")

    # cap = cv2.VideoCapture('father.mp4')
    # test_frames = []
    # success, image = cap.read()
    # while success:
    #     image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
    #     test_frames.append(image)
    #     success, image = cap.read()
    # cap.release()

    # x = np.array(test_frames)
    # x_features = base_model.predict(x)
    # answer = model.predict(x_features)
    # print(answer)
