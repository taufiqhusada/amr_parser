import pandas as pd
from stog.data.dataset_readers.amr_parsing.io import AMRIO
from utils.amr import dump_amr_features

def create_dependency_parser_feature(annotation, sentence, sentence_id=0):

    dependency_parser_feature = {
        'sentence_id': [],
        'parent': [], 
        'parent_position': [], 
        'child': [], 
        'child_position': [], 
        'is_root' : [],
        'sequence': [],
        'parent_ner': [],
        'child_ner': [],
        'parent_pos': [],
        'dependency_role': [],
        'child_pos': []
    }

    for idx, dependency in enumerate(annotation['dependency']):
        head = dependency['head']
        if (head != 0):
            head_idx = head-1
            dependency_parser_feature['sentence_id'].append(sentence_id)
            dependency_parser_feature['sequence'].append(idx+1)
            dependency_parser_feature['parent'].append(annotation['lemmas'][head_idx])
            dependency_parser_feature['parent_position'].append(head_idx)
            dependency_parser_feature['child'].append(annotation['lemmas'][idx])
            dependency_parser_feature['child_position'].append(annotation['lemmas'].index(annotation['lemmas'][idx]))
            dependency_parser_feature['is_root'].append(1 if annotation['dependency'][head_idx]['head'] == 0 else 0)
            dependency_parser_feature['parent_ner'].append(annotation['ner_tags'][head_idx])
            dependency_parser_feature['child_ner'].append(annotation['ner_tags'][idx])
            dependency_parser_feature['parent_pos'].append(annotation['pos_tags'][head_idx])
            dependency_parser_feature['child_pos'].append(annotation['pos_tags'][idx])
            dependency_parser_feature['dependency_role'].append(dependency['relation'])
            
    return dependency_parser_feature

def create_dependency_parser_feature_from_sentence(annotator, sentence):
    annotation = annotator.annotate(sentence)
    return create_dependency_parser_feature(annotation, sentence)

def create_dependency_parser_feature_from_file(annotator, filepath):
    dependency_feature_data = []
    amrs = []
    sentence_ids = []
    
    with open(filepath + '.features', 'w', encoding='utf-8') as f:
        for i, amr in enumerate(AMRIO.read(filepath), 1):
            if i % 100 == 0:
                print('{} processed.'.format(i))

            annotation = annotator.annotate(amr.sentence)
            dump_amr_features(amr, annotation, f)
            sentence_data = create_dependency_parser_feature(annotation, amr.sentence, i)
            dependency_feature_data.append(sentence_data)
            sentence_ids.append(i)
            amrs.append(amr)

    dataset_dict = {
        'sentence_id': sum([sum([sentence_data['sentence_id'] for sentence_data in dependency_feature_data], [])],[]),
        'sequence': sum([sum([sentence_data['sequence'] for sentence_data in dependency_feature_data], [])],[]),
        'parent': sum([sum([sentence_data['parent'] for sentence_data in dependency_feature_data], [])],[]),
        'parent_position': sum([sum([sentence_data['parent_position'] for sentence_data in dependency_feature_data], [])],[]), 
        'child': sum([sum([sentence_data['child'] for sentence_data in dependency_feature_data], [])],[]), 
        'child_position': sum([sum([sentence_data['child_position'] for sentence_data in dependency_feature_data], [])],[]), 
        'is_root' : sum([sum([sentence_data['is_root'] for sentence_data in dependency_feature_data], [])],[]),
        'parent_ner': sum([sum([sentence_data['parent_ner'] for sentence_data in dependency_feature_data], [])],[]),
        'child_ner': sum([sum([sentence_data['child_ner'] for sentence_data in dependency_feature_data], [])],[]),
        'parent_pos': sum([sum([sentence_data['parent_pos'] for sentence_data in dependency_feature_data], [])],[]),
        'dependency_role': sum([sum([sentence_data['dependency_role'] for sentence_data in dependency_feature_data], [])],[]),
        'child_pos': sum([sum([sentence_data['child_pos'] for sentence_data in dependency_feature_data], [])],[])
    }

    amr_dict = {
        'sentence_id': sentence_ids,
        'amr': [str(amr.graph) for amr in amrs]
    }

    dependency_feature_df = pd.DataFrame(dataset_dict)
    amr_df = pd.DataFrame(amr_dict)
    
    return dependency_feature_df, amr_df
