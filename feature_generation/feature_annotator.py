import nltk
import stanza
import string

from collections import defaultdict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from utils.ner.entity_recognizer import get_entities

class FeatureAnnotator:
    def __init__(self):
        self.nlp = stanza.Pipeline("id",use_gpu=False)
        self.stemmer = StemmerFactory().create_stemmer()
        self.ner = get_entities
        # Set POS Tagger 
        self.pos_tagger = nltk.tag.CRFTagger()
        self.pos_tagger.set_model_file('pretrained/pos_tagger/all_indo_man_tag_corpus_model.crf.tagger')

    def annotate(self, sentence):
        annotation = defaultdict(list)
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        doc = self.nlp(sentence)
        
        annotation['ner_tags'] = self.ner(sentence)
        
        word_dict = defaultdict(int)
        
        for sent in doc.sentences:
            for idx, word in enumerate(sent.words):
                annotation['tokens'].append(word.text)
                stemmed_word = self.stemmer.stem(word.text)                
                if (annotation['ner_tags'][idx] in ['PER','ORG']):
                    stemmed_word = word.text.lower()
                annotation['lemmas'].append(stemmed_word+'_{}'.format(word_dict[stemmed_word]))
                annotation['dependency'].append(dict(relation=word.deprel, head=word.head))
        
        annotation['pos_tags'] = [tag[1] for tag in self.pos_tagger.tag(annotation['tokens'])]
                    
        return annotation
