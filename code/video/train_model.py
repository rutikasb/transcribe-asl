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

# BATCH_SIZE = 32
# SEED = 42

def extract_cnn_features(raw_data_dir, features_dir):
    # base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # print("Base model shape")
    # print(base_model.input_shape)

    model = InceptionV3(weights='imagenet', include_top=True)
    base_model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    _, height, width, channels = base_model.input_shape

    frames = FramesSeqGenerator(raw_data_dir, BATCH_SIZE, 20, height, width, channels, False)
    for i, video in frames.videos_list.iterrows():
        video_name = video.frames_dir.split('/')[-1]
        label = video.label
        print(f'Video label = {label}')
        features_path = f'{features_dir}/{label}/{video_name}.npy'
        X, y = frames.generate(video)
        print(f'X shape = {X.shape}')
        features = base_model.predict(X)
        print(f'features shape = {features.shape}')

        if not os.path.exists(f'{features_dir}/{label}'):
            os.makedirs(f'{features_dir}/{label}')
        np.save(features_path, features)
    return

def extract_cnn_features_all(raw_data_dir, features_dir):
    model = InceptionV3(weights='imagenet', include_top=True)
    base_model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    _, height, width, channels = base_model.input_shape

    print(base_model.input_shape)

    signs = sorted([name for name in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, name))])
    for sign in signs:
        print("\nSign = {0}\n-------------------".format(sign))
        videos = sorted([name for name in os.listdir(os.path.join(raw_data_dir, sign)) if os.path.isdir(os.path.join(raw_data_dir, sign, name))])
        
        # For each video we will save the cnn features computed on each image
        for video in videos:
            print("Video = {0}".format(video))
            features_path = f'{features_dir}/{sign}/{video}.npy'
            if not os.path.exists(f'{features_dir}/{sign}'):
                os.makedirs(f'{features_dir}/{sign}')

            pathname = os.path.join(raw_data_dir, sign, video, '*.jpg')
            imagenames = sorted(glob.glob(pathname))
            featuresList = []
            for imagename in imagenames:
                img = image.load_img(imagename, target_size=(height, width))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                inceptionv3_feature = base_model.predict(img_data)
                featuresList.append(inceptionv3_feature)
            stackedFeatures = np.vstack(featuresList)
            np.save(features_path, stackedFeatures)
            print("{0} images processed\n".format(len(featuresList)))



def train_lstm(features_path, epochs, num_features, num_classess):
    model = Sequential()
    model.add(LSTM(num_features, dropout=0.2, input_shape=(20, num_features), return_sequences=True))
    model.add(LSTM(num_features * 1, return_sequences=False))
    model.add(Dense(num_classess, activation='softmax'))
    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0)]
    callbacks = [ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0)]

    train_features = FeaturesSeqGenerator(features_path + '/train', BATCH_SIZE, model.input_shape[1:])
    val_features = FeaturesSeqGenerator(features_path + '/test', BATCH_SIZE, model.input_shape[1:])
    model.fit_generator(generator=train_features,
            validation_data=val_features,
            epochs=int(epochs),
            workers=1,
            use_multiprocessing=False,
            verbose=1,
            callbacks=callbacks)
    return model

def train_lstm_jittered(features_path, sequencelength, epochs, num_features, num_classess):
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False,input_shape=(sequencelength, num_features),dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classess, activation='softmax'))
    optimizer = Adam(lr=1e-4)
    metrics = ['accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    callbacks = [ModelCheckpoint('video_LSTM.h5', monitor='val_loss', save_best_only=True, verbose=1)]

    data_path, batch_size, featureLength, shuffle = 'cnn_features/train', 32, 2048, True
    train_gen = DataGenerator(data_path,
                            batch_size=batch_size,
                            featureLength=featureLength,
                            seqLength=sequencelength,
                            shuffle=shuffle)

    data_path, batch_size, featureLength, shuffle = 'cnn_features/test', 16, 2048, False
    val_gen = DataGenerator(data_path,
                            batch_size=batch_size,
                            featureLength=featureLength,
                            seqLength=sequencelength,
                            shuffle=shuffle)
    
    model.fit_generator(generator=train_gen,
            validation_data=val_gen,
            epochs=int(epochs),
            workers=1,
            use_multiprocessing=False,
            verbose=1,
            callbacks=callbacks)
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN+LSTM model for short video snippets')

    parser.add_argument('--data-path', default='processed_data', required=False, help='path to the train and test directories')
    parser.add_argument('--lstm-epochs', default=100, required=False, help='number of LSTM epochs')
    parser.add_argument('--reload-cnn-features', default=True, required=False, help='re-extract CNN features from frames')
    parser.add_argument('--features-path', default='cnn_features', required=False, help='extracted CNN features path')
    args = vars(parser.parse_args())
    print(args)
    if args.get('data_path') is None:
        print('Please provide path to video frames')
        exit()

    # GPU configuration
    # K.tensorflow_backend._get_available_gpus()
    # config = tf.ConfigProto( device_count = {'GPU': 200, 'CPU': 4} )
    # sess = tf.Session(config=config)
    # keras.backend.set_session(sess)

    # extract_cnn_features(f'{args["data_path"]}/train', f'{args["features_path"]}/train')
    # extract_cnn_features(f'{args["data_path"]}/test', f'{args["features_path"]}/test')

    # extract_cnn_features_all(f'{args["data_path"]}/train', f'{args["features_path"]}/train')
    # extract_cnn_features_all(f'{args["data_path"]}/test', f'{args["features_path"]}/test')

    # model = train_lstm(args['features_path'], args['lstm_epochs'], num_features, num_classes)

    num_features = 2048
    num_classes = len(os.listdir(f'{args["features_path"]}/train'))
    seqLength = 10
    model = train_lstm_jittered(args['features_path'], seqLength, args['lstm_epochs'], num_features, num_classes)

    # model.save("final_model.h5")
