from tqdm import tqdm
from constants.model import K_SPLIT
from training.model import ExperimentParams, run_k_fold_validation
from training.encoding import one_hot_encode, label_encode
from feature_generation.feature_annotator import FeatureAnnotator
from feature_generation.dependency_pair import create_dependency_parser_feature_from_file
from feature_generation.amr_pair import get_amr_data, create_pair_labels, create_labels

from preprocessing.model_feature import filter_pair
from utils.word_vec import load_word_vec

def init_experiment_params():
    experiment_params = []
    experiment_params.append(ExperimentParams(model_name='xgb', model_params=[
    #     {"learning_rate" : 0.05, 'max_depth': 5},
    #     {"learning_rate" : 0.05, 'max_depth': 8 },
    #     {"learning_rate" : 0.05, 'max_depth': 10},
    #     {"learning_rate" : 0.10, 'max_depth': 5 },
        {"learning_rate" : 0.10, 'max_depth': 8},
    #     {"learning_rate" : 0.10, 'max_depth': 10},
    #     {"learning_rate" : 0.20, 'max_depth': 5},
    #     {"learning_rate" : 0.20, 'max_depth': 8},
    #     {"learning_rate" : 0.20, 'max_depth': 10},
    ]))

    # experiment_params.append(ExperimentParams(model_name='dtc', model_params=[
    #     {'max_depth': 6 , 'criterion': 'entropy'},
    #     {'max_depth': 6 , 'criterion': 'gini'},
    #     {'max_depth': 7 , 'criterion': 'entropy'},
    #     {'max_depth': 7 , 'criterion': 'gini'}, 
    #     {'max_depth': 10 , 'criterion': 'entropy'},
    #     {'max_depth': 10 , 'criterion': 'gini'},
    #     {'max_depth': 12 , 'criterion': 'entropy'},
    #     {'max_depth': 12 , 'criterion': 'gini'}
    # ]))

    return experiment_params


def run_training(experiment_params, dependency_features_df, word_vec, one_hot_encoder, label_encoder):
    trained_models = []
    experiment_tqdm = tqdm(experiment_params)
    
    for experiment_param in experiment_tqdm:
        train_tqdm = tqdm(experiment_param.model_params)
        for model_param in train_tqdm: 
            model, performance = run_k_fold_validation(dependency_features_df,
                                                    n_split=K_SPLIT,
                                                    shuffle=True,
                                                    word_vec=word_vec,
                                                    label_encoder=label_encoder,
                                                    one_hot_encoder=one_hot_encoder,
                                                    feature_filter=feature_filter_type,
                                                    model_name=experiment_param.model_name, 
                                                    params=model_param)
            model_param_string = ''
            if (model_param is not None):
                model_param_string = '_'.join(["{}_{}".format(key, model_param[key]) for key in model_param])
            model_name = experiment_param.model_name + '_' + model_param_string
            trained_models.append(TrainingModel(model_name, model, performance, labels))

    return trained_models

def train(filepath):
    annotator = FeatureAnnotator()
    dependency_parser_feature_df, amr_df = create_dependency_parser_feature_from_file(annotator, filepath)
    filtered_dependency_parser_feature_df = filter_pair(dependency_parser_feature_df)
    print(filtered_dependency_parser_feature_df.head())
    amr_data = get_amr_data(filepath + '.features')
    pair_labels = create_pair_labels(amr_data)
    labels, unk_pairs = create_labels(pair_labels,filtered_dependency_parser_feature_df)
    # combine label and dependency parser feature
    dependency_features_dict = filtered_sentence_dataset.to_dict()
    dependency_features_dict['label'] = pd.Series(labels)
    dependency_features_df = pd.DataFrame(dependency_features_dict)
    # load encoders
    one_hot_encoder = one_hot_encode(dependency_features_df)
    label_encode = label_encode(dependency_features_df)

    # run experiments
    experiments = init_experiment_params()
    trained_models = run_training(experiments, dependency_features_df)
    
    best_model = sorted(trained_models, reverse=True)[0]
    best_model.print_metrics()
    best_model.save()
