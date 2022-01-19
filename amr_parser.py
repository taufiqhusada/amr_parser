import sys
from utils.argument_parser import parse_arguments
import warnings
warnings.filterwarnings("ignore")

args = parse_arguments()

from training.train import train
from prediction.predict import predict_from_sentence, predict_from_file
from postprocess.amr_graph import save_amr_graphs

if __name__ == "__main__":
    filepath = args.file
    output_path = args.output

    if args.server: 
        from api.rest import app
        app.run(threaded=False)
    else:
        if args.predict:
            sentence = args.sentence
            if sentence is not None and filepath is not None:
                print("Choose either to predict a sentence or a file")
                sys.exit(1)
            elif sentence is not None:
                amr_graph = predict_from_sentence(sentence)
                save_amr_graphs([(sentence, amr_graph)], output_path)

            else:
                list_pair_sentence_amr = predict_from_file(filepath)
                save_amr_graphs(list_pair_sentence_amr, output_path)

        if args.train:
            if filepath is None:
                print("Specify training data filepath")
                sys.exit(1)
            else:
                train(filepath)
