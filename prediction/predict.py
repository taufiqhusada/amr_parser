import pandas as pd 

from feature_generation.feature_annotator import FeatureAnnotator
from constants.paths import PRETRAINED_BEST_MODEL_PATH
from constants.model import PAIR_FILTER_TYPE, FEATURE_FILTER_TYPE
from utils.model import load_model, load_word_vec, load_one_hot_encoder, load_label_encoder
# Feature generations
from feature_generation.dependency_pair import create_dependency_parser_feature_from_sentence
from preprocessing.model_feature import feature_selection, preprocess, filter_pair
# Post process
from postprocess.amr_graph import create_pediction_dict, create_amr_graph_from_prediction

annotator = FeatureAnnotator()
model = load_model(PRETRAINED_BEST_MODEL_PATH)
word_vec = load_word_vec()
one_hot_encoder = load_one_hot_encoder()
label_encoder = load_label_encoder()

def process_sentence(sentence, sentence_id=0):
    dependency_parser_feature = create_dependency_parser_feature_from_sentence(annotator, sentence)
    X = pd.DataFrame(dependency_parser_feature)
    filtered_X = filter_pair(X, PAIR_FILTER_TYPE)
    return filtered_X

def predict_and_process(filtered_X):
    selected_X = feature_selection(filtered_X, FEATURE_FILTER_TYPE)
    X_preprocessed = preprocess(selected_X, word_vec, one_hot_encoder)
    y_pred = model.predict(X_preprocessed)

    # Post process
    prediction = create_pediction_dict(filtered_X, y_pred, label_encoder)[0]
    amr_graph = create_amr_graph_from_prediction(prediction)
    return amr_graph


def predict_sentence(sentence, sentence_id=0):
    dependency_parser_feature = create_dependency_parser_feature_from_sentence(annotator, sentence)
    X = pd.DataFrame(dependency_parser_feature)
    filtered_X = filter_pair(X, PAIR_FILTER_TYPE)

    # preprocess X    
    selected_X = feature_selection(filtered_X, FEATURE_FILTER_TYPE)
    X_preprocessed = preprocess(selected_X, word_vec, one_hot_encoder)

    y_pred = model.predict(X_preprocessed)

    # Post process
    prediction = create_pediction_dict(filtered_X, y_pred, label_encoder)[0]
    amr_graph = create_amr_graph_from_prediction(prediction)

    return amr_graph

def predict_from_sentence(sentence):
    amr_graph = predict_sentence(sentence)
    return amr_graph

def predict_from_file(filepath):
    amr_graphs = []
    sentences = []

    with open(filepath, 'r') as input_file:
        lines = input_file.readlines()
        lines = filter(lambda x: x != '\n', lines)
        for line in lines:
            if len(line) > 0 and line is not None and len(line.split('.')) == 2:
                print(line)
                sentences.append(line)
                amr_graph = predict_sentence(line)
                amr_graphs.append(amr_graph)

    return sentences, amr_graphs
