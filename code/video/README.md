#### Download raw videos for signs in train and test sets
```
python code/s3_download.py
```

#### Convert the raw videos into frames
```
python code/video/convert_to_frames.py --raw-data-path raw_data --processed-data-path processed_data
```

#### Optionally convert the frames to optical flow images
```
python code/video/optical_flow.py --data-path processed_data/ --output-path optical_flow
```

#### Train the model on frames
```
python code/video/train_model.py --data-path processed_data --lstm-epochs 10
```

#### Train the model on optical flow images
```
python code/video/train_model.py --data-path optical_flow --lstm-epochs 10
```
