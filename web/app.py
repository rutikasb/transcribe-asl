import io, os, traceback
import subprocess
import json
from flask import Flask, Blueprint
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import reqparse
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(app)

UPLOAD_KEY = 'image'
UPLOAD_LOCATION = 'files'
ns = api.namespace('/', description='ASL predictor')
upload_parser = api.parser()
upload_parser.add_argument(UPLOAD_KEY,
                           location=UPLOAD_LOCATION,
                           type=FileStorage,
                           required=True)

@ns.route('/predict')
class PredictSigns(Resource):
    @ns.expect(upload_parser)
    def post(self):
        try:
            image_file = request.files[UPLOAD_KEY]
            image_file.save("image_to_predict.jpg")
            image = io.BytesIO(image_file.read())
            # TODO; Change this to final model's prediction script
            subprocess.call("python -m scripts.label_image --graph=model/retrained_graph.pb --image=sunflower-bunch_800x.jpg --labels=model/retrained_lables.txt", shell=True)
            with open('output.json', 'r') as f:
                data = json.load(f)
            return data
        except Exception as ex:
            traceback.print_exc()
            return {'message': 'something wrong with incoming request. ' +
                               'Original message: {}'.format(ex)}, 400

if __name__ == '__main__':
    blueprint = Blueprint('', __name__, url_prefix='/')
    api.init_app(blueprint)
    app.register_blueprint(blueprint)
    app.run(debug=True)
