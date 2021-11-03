import json
import base64
from flask import Flask
from flask import request, jsonify, make_response
from flask_cors import CORS
from prediction.predict import predict_and_process, process_sentence
from utils.amr import penman_to_dot

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET'])
def predict_sentence():
    path = "api/output/amr"
    sentence = request.args.get('sentence')
    try:
        filtered_X = process_sentence(sentence)
        amr_graph = predict_and_process(filtered_X)
    except Exception as e:
        print(e)
        return make_response("Error", 400)
    features = json.dumps(list(filtered_X.T.to_dict().values()))
    response = {}
    response['amr_graph'] = str(amr_graph)
    response['triples'] = amr_graph._triples
    response['edges'] = amr_graph.edges()
    response['top'] = amr_graph._top
    response['features'] = json.loads(features)


    response['dot'] = penman_to_dot(amr_graph, path)
    with open("{}.png".format(path), "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read())
        response['img'] = encoded_img.decode('utf-8')
    
    return jsonify(response)
