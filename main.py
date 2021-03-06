#from generate_walk import MetaPathGenerator
#from preprocessing import preprocess_
from pder import PDER

import os, sys
from optparse import OptionParser

# :os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def runPDER(options):

    # Check validity of parameters
#     if type(options.preprocess) is not bool:
#         print("Invalid -p value. Should be \"True\" or \"False\"",
#               file=sys.stderr)
#         sys.exit()

#     if options.preprocess:
#         preprocess_(dataset=options.dataset,
#                     threshold=options.test_threshold,
#                     prop_test=options.proportion_test,
#                     sample_size=options.test_size)

#     # preprocessing
#     if options.gen_mp:
#         mp_generator = MetaPathGenerator(
#             length=options.length,
#             coverage=options.coverage,
#             dataset=options.dataset)
#         walks = mp_generator.generate_metapaths(
#             patterns=options.meta_paths.split(" "),
#             alpha=options.alpha)
#         mp_generator.write_metapaths(walks)

    # init data_loader
    pder_model = PDER(
        dataset=options.dataset,
        embedding_dim=options.embedding_dim,
        epoch_num=options.epoch_num,
        batch_size=options.batch_size,
        # window_size=options.window_size,
        neg_sample_ratio=options.neg_ratio,
        lstm_layers=options.lstm_layers,
        include_content=options.include_content,
        lr=options.learning_rate,
        cnn_channel=options.cnn_channel,
        lambda_=options.lambda_,
        prec_k=options.prec_k,
        test_ratio=options.test_ratio,
        # test_prop=options.proportion_test,
        # neg_test_ratio=options.neg_test_ratio,
        mp_length=options.length,
        mp_coverage=options.coverage,
        id=options.id,
        answer_sample_ratio=options.answer_sample_ratio
    )

    pder_model.run()
    pder_model.test()



if __name__ == '__main__':
    """Generating random walks and output to file
    
    [a, b, c, d, e, f, g, h, i, [j], k, 
     l, m, n, o, p, q, r, [s], t, u, v,
     [w], x, y, z]

    Args:
        -d, --dataset (str)
        -l, --length (int)
        -c, --coverage (int)
        -a, --alpha (int)
        -m, --meta-paths (str, split by " ")
        -p, --preprocess (bool)
        -w, --window-size (size)
        -g, --gen-metapaths (bool)
        -n, --neg-ratio (float)
        -e, --embedding-dim (int)
        -y, --lstm-layers (int)
        -o, --epoch-number (int)
        -b, --batch-size (int)
        -u, --include-content (bool)
        -r, --learning-rate (float)
        -t, --test-threshold (int)
        -f, --proportion-test (float)
        -v, --cnn-channel (int)
        -j, --answer_sample_ratio

    Returns:
        do everything

    """

    parser = OptionParser()
    parser.add_option("-d", "--dataset", type="string",
                      dest="dataset", default="3dprinting",
                      help="The dataset to work on.")

    parser.add_option("-l", "--length", type="int",
                      dest="length", default=15,
                      help="The length of the random walk to be generated.")

    parser.add_option("-c", "--coverage", type="int",
                      dest="coverage", default=2,
                      help="The number of times each node to be covered.")

    parser.add_option("-a", "--alpha", type="float",
                      dest="alpha", default=0.0,
                      help="The probability of restarting in meta-path generating")

    parser.add_option("-m", "--metapaths", type="string",
                      dest="meta_paths",
                      help="The target meta-paths used to generate the data file, "
                           "split by space, enclose by \"\".")

    parser.add_option("-p", "--preprocess", default=False,
                      dest="preprocess", action="store_true",
                      help="Adding it to indicate doing preprocessing.")

    parser.add_option("-w", "--window-size", type="int",
                      dest="window_size", default=5,
                      help="The window size of the meta-path model.")

    parser.add_option("-g", "--gen-metapaths", default=False,
                      dest="gen_mp", action="store_true",
                      help="Decide whether to generate new metapaths.")

    parser.add_option("-n", "--neg-ratio", type="float",
                      dest="neg_ratio", default=1.2,
                      help="The ratio of negative samples.")

    parser.add_option("-e", "--embedding-dim", type="int",
                      dest="embedding_dim", default=300,
                      help="The embedding dimension of the model.")

    parser.add_option("-y", "--lstm-layers", type="int",
                      dest="lstm_layers", default=3,
                      help="The number of layers of the LSTM model.")

    parser.add_option("-o", "--epoch-number", type="int",
                      dest="epoch_num", default=1000,
                      help="The epoch number of the training set.")

    parser.add_option("-b", "--batch-size", type="int",
                      dest="batch_size", default=10,
                      help="The number of meta-paths fed into the model each batch")

    parser.add_option("-u", "--include-content", default=False,
                      dest="include_content", action="store_true",
                      help="Whether to include content in the text embedding")

    parser.add_option("-r", "--learning-rate", type="float",
                      dest="learning_rate", default=0.01,
                      help="The learning rate.")

    parser.add_option("-t", "--test-threshold", type="int",
                      dest="test_threshold", default=3,
                      help="The threshold of an instant selected as test")

    parser.add_option("-f", "--proportion-test", type="float",
                      dest="proportion_test", default=0.1,
                      help="The proportion of the test dataset.")

    parser.add_option("-v", "--cnn-channel", type="int",
                      dest="cnn_channel", default=32,
                      help="How many channels for CNN intermediate output.")

    parser.add_option("-z", "--lambda", type="float",
                      dest="lambda_", default=1.0,
                      help="The hyperparam between NE loss and Rank loss.")

    parser.add_option("-k", "--precision_at_K", type="int",
                      dest="prec_k", default=3,
                      help="The hyperparam to test Precision@K.")
    
    parser.add_option("-x", "--test-size", type="int",
                      dest="test_size", default=2,
                      help="Random selected sample numbers in test batch")

    parser.add_option("-i", "--id-number", type="int",
                      dest="id", default=0,
                      help="The identifier of a unique experiment")

    parser.add_option("-q", "--test-ratio", type="float",
                      dest="test_ratio", default=0.05,
                      help="The ratio of test dataset used in the validation set")

    parser.add_option("-j", "--answer-sample-ratio", type="float",
                      dest="answer_sample_ratio", default=0.5,
                      help="The ratio of sample answer")


    (options, args) = parser.parse_args()

    runPDER(options)
