## Transcribe American Sign Language

American Sign Language (ASL) is the third most taught language in the US with ~150,000 currently enrolled students, and is a promary language of more than 1 million people.

## About this code

This code base is for training a CNN and LSTM based model which can classify the different signs of ASL. Give a short video snippet of a person doing an ASL sign, the model should be able to tell what sign it i.e. convert sign language to english text.

The code is developed on Python 3.6.

## How to run

### Get the dataset
The dataset used here is the American Sign Language Lexicon Video Dataset (ASLLVD) circo 2008 which is available on the Boston University website. The model presented here was trained on just 15 signs out of th more than 3k signs made available by the ASLLVD. The number of videos available in this dataset for each sign, however is very small. It ranges from 1 to 6 videos. This was not sufficient for our training and we recorded our videos in the process and was one of the main reasons to limit the number of signs the model was trained on to just 15.

We haven't made the self recorded videos public, but you get the BU videos we used with the following command
`python code/s3_download.py`

### Training

#### Convert raw videos to frames
`python code/video/convert_to_frames.py --raw-data-path raw_data --processed-data-path processed_data`

#### Train the model 
`python code/video/train_model.py --data-path processed_data --lstm-epochs 10`


## Pre-trained model
You can download a pre-trained model which can recognize the signs AGAIN, BEAUTIFUL, BOY, CAR, DRINK, FAMILY, HELLO, NAME, WALK from [here](https://drive.google.com/file/d/1Zr4YToaHmilSioaKwm0iyVh0CzoTdc3H/view?usp=sharing).
