import anago
from anago.utils import load_data_and_labels, load_glove

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import pickle
import numpy as np

MODEL_DIRPATH = "pretrained/ner_tagger/"
WEIGHT_FILENAME = "nerweight.h5"
PARAMS_FILENAME = "nerparams.json"
PREPROCESSOR_FILENAME = "nerprepro.pkl"

MODEL_NAME = "model_ner_12514_softmax_v5_w2v_100_POS_LSTM_EmbNotTrainable_OOV"
loaded_model = load_model(f"pretrained/{MODEL_NAME}")

with open(f"pretrained/{MODEL_NAME}/dict.pickle", 'rb') as handle:
    idx2token_loaded = pickle.load(handle)
    idx2tag_loaded = pickle.load(handle)
    token2idx_loaded = pickle.load(handle)
    tag2idx_loaded = pickle.load(handle)

model = anago.Sequence().load(weights_file=MODEL_DIRPATH + WEIGHT_FILENAME, 
            params_file=MODEL_DIRPATH + PARAMS_FILENAME, 
            preprocessor_file=MODEL_DIRPATH + PREPROCESSOR_FILENAME)

def __get_ner_tags(result):
    ner_tags = ["O" for _ in result['words']]
    for entity in result['entities']:
        ner_tags[entity['beginOffset']] = entity['type']
    return ner_tags

def get_entities(sentence):
    result = model.analyze(sentence)
    return __get_ner_tags(result)

def get_entities_tf(sentence,  pad_token = np.nan, n_timesteps = 48): 
    sentence=sentence.split()
    padded_sentence = sentence + [pad_token] * (n_timesteps - len(sentence))
    padded_sentence = [token2idx_loaded.get(w, 0) for w in padded_sentence]

    pred = loaded_model.predict(np.array([padded_sentence]))
    # pred
    ner_tags = []
    for i in range(len(sentence)):
      pred_idx = np.argmax(pred[0][i], axis=-1)
      pred_tag = idx2tag_loaded[pred_idx].split("-")[-1]
      ner_tags.append(pred_tag)

    return ner_tags
