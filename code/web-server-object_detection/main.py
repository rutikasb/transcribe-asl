from threading import Thread
import numpy as np
import tensorflow as tf
import base64
import flask
import uuid
import time
import json
import sys
import io
import logging
import cv2
from PIL import Image
from keras.preprocessing.image import img_to_array


IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 1
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_PATH = './models/exported_graphs'
PATH_TO_FROZEN_GRAPH = MODEL_PATH + '/frozen_inference_graph.pb'
global detection_graph

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)


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
	    print("Loaded {0}".format(PATH_TO_FROZEN_GRAPH))

loadModel()

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


def processImage(img, conf_thresh):
	global detection_graph
	img = img.astype(np.uint8)
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	height, width, channels = img.shape
	print("height={0}, width={1}, channels={2}".format(height, width, channels))
	image_np_expanded = np.expand_dims(img, axis=0)
	output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
	det_boxes = output_dict['detection_boxes']
	det_classes = output_dict['detection_classes']
	det_score = output_dict['detection_scores']
	r=0
	result = []
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
	        result.append({'label': label, 'conf': float(score)})
	    r += 1

	return result


def prepare_image2(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
 
	# resize the input image and preprocess it
	image = img_to_array(image)
	imageBGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	print("type of image: {0}, shape of image: {1}".format(type(imageBGR), imageBGR.shape))

 
	# return the processed image
	return imageBGR

def classify_process():
	loadModel()


	# continually poll for new images to classify
	while True:
		# attempt to grab a batch of images from the database, then
		# initialize the image IDs and batch of images themselves
		queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
		imageIDs = []
		batch = []

		# loop over the queue
		for q in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
			# image = base64_decode_image(q["image"], IMAGE_DTYPE,(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
			image = base64_decode_image(q["image"], IMAGE_DTYPE, q["shape"])
			print("Shape of image after base64_decode_image: {}".format(image.shape))


			imageIDs.append(q["id"])
			batch.append(image)

		for im,imID in zip(batch, imageIDs):
			result = processImage(im, 0.1)
			db.set(imID, json.dumps(result))
		db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

		# if len(imageIDs) > 0:
		# 	# classify the batch
		# 	print("* Batch size: {}".format(batch.shape))
		# 	result = processImage(batch, 0.1)
		# 	for imageID in imageIDs:
		# 		print(type(result))
		# 		print(result)
		# 		db.set(imageID, json.dumps(result))
		# 	db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

		# sleep for a small amount
		time.sleep(SERVER_SLEEP)





@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
 
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# classification
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			print("type of image: {0}".format(type(image)))
			image = prepare_image2(image)
			data["predictions"] = processImage(image, 0.1)
 
			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			# image = image.copy(order="C")
 
			# # generate an ID for the classification then add the
			# # classification ID + image to the queue
			# k = str(uuid.uuid4())
			# d = {"id": k, "image": base64_encode_image(image), "shape":image.shape}
			# db.rpush(IMAGE_QUEUE, json.dumps(d))

			# # keep looping until our model server returns the output
			# # predictions
			# while True:
			# 	# attempt to grab the output predictions
			# 	output = db.get(k)
 
			# 	# check to see if our model has classified the input
			# 	# image
			# 	if output is not None:
 		# 			# add the output predictions to our data
 		# 			# dictionary so we can return it to the client
			# 		output = output.decode("utf-8")
			# 		data["predictions"] = json.loads(output)
 
			# 		# delete the result from the database and break
			# 		# from the polling loop
			# 		db.delete(k)
			# 		break
 
			# 	# sleep for a small amount to give the model a chance
			# 	# to classify the input image
			# 	time.sleep(CLIENT_SLEEP)
 
			# indicate that the request was a success
			data["success"] = True
 
	# return the data dictionary as a JSON response
	return flask.jsonify(data)






# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	# load the function used to classify input images in a *separate*
	# thread than the one used for main classification
	# print("* Starting model service...")
	# t = Thread(target=classify_process, args=())
	# t.daemon = True
	# t.start()

	# start the web server
	print("* Starting web service...")
	loadModel()
	# This is used when running locally. Gunicorn is used to run the
	# application on Google App Engine. See entrypoint in app.yaml.
	app.run(host='127.0.0.1', port=8080, debug=True)