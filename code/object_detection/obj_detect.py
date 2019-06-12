import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
import cv2
import argparse


# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_PATH = './models/exported_graphs'
PATH_TO_FROZEN_GRAPH = MODEL_PATH + '/frozen_inference_graph.pb'
global detection_graph

#Load Tensorflow frozen model
def loadModel():
	global detection_graph
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')
	    # print("Loaded {0}".format(PATH_TO_FROZEN_GRAPH))

label_map = {}
label_map[1] = 'above'
label_map[2] = 'accept'
label_map[3] = 'accident'
label_map[4] = 'add'
label_map[5] = 'afraid'
label_map[6] = 'airplane'
label_map[7] = 'alarm'
label_map[8] = 'apple'
label_map[9] = 'arrive'
label_map[10] = 'a'
label_map[11] = 'enter'
label_map[12] = 'lock'
label_map[13] = 'million'
label_map[14] = 'old'
label_map[15] = 'one'


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

def processImage(filename, conf_thresh):
	global detection_graph
	img = cv2.imread(filename, cv2.IMREAD_COLOR)
	height, width, channels = img.shape
	image_np_expanded = np.expand_dims(img, axis=0)
	output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
	det_boxes = output_dict['detection_boxes']
	det_classes = output_dict['detection_classes']
	det_score = output_dict['detection_scores']
	r=0
	for score in np.nditer(det_score):
	    if(score > conf_thresh):
	        ymin,xmin,ymax,xmax = tuple(det_boxes[r])
	        xmin = int(width*xmin)
	        xmax = int(width*xmax)
	        ymin = int(height*ymin)
	        ymax = int(height*ymax)
	        # label = category_index[det_classes[r]]['name']
	        label = label_map[det_classes[r]]
	        print("label={0}, conf={1}".format(label, score))
	    r += 1


def processVideo(filename, conf_thresh):
	global detection_graph
	# print("Processing video file: {0}".format(filename))
	cap = cv2.VideoCapture(filename)
	total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))

	frameNum = 0
	startFrame = 0
	cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
	while(cap.isOpened()):
	    ret, frame = cap.read()
	    if ret == False:
	    	break
	    if cv2.waitKey(10) & 0xFF == ord('q'):
	        break
	    frameNum += 1
	    image_np_expanded = np.expand_dims(frame, axis=0)
	    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
	    det_boxes = output_dict['detection_boxes']
	    det_classes = output_dict['detection_classes']
	    det_score = output_dict['detection_scores']
	    r=0
	    for score in np.nditer(det_score):
	        if(score > conf_thresh):
	            ymin,xmin,ymax,xmax = tuple(det_boxes[r])
	            xmin = int(width*xmin)
	            xmax = int(width*xmax)
	            ymin = int(height*ymin)
	            ymax = int(height*ymax)
	            # label = category_index[det_classes[r]]['name']
	            label = label_map[det_classes[r]]
	            print("frameNum={0}, label={1}, conf={2}".format(frameNum, label, score))
	        r += 1
	            # cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),3)
	            # font = cv2.FONT_HERSHEY_SIMPLEX
	            # cv2.putText(frame,label,(xmin-5,ymin-5), font,1,(255,255,255),2,cv2.LINE_AA)
	    # cv2.imshow('frame',frame)

	cap.release()
	# cv2.destroyAllWindows()



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process image file or video file')
	parser.add_argument("-i", "--image", required=False, help="path to image file")
	parser.add_argument("-v", "--video", required=False, help="path to video file")
	args = vars(parser.parse_args())
	if args['image'] is None and args['video'] is None:
		print("Need to provide an image file or video file")
	else:
		loadModel()
		conf_thresh = 0.1
		if args['image'] is not None:
			processImage(args['image'], conf_thresh)
		else:
			processVideo(args['video'], conf_thresh)


