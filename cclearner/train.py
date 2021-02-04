from common import common
from config import Config
from extractor import Extractor
import pandas as pd

JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
SHOW_TOP_CONTEXTS = 10


def train(config: Config, model):
    path_extractor = Extractor(config,
                          jar_path=JAR_PATH,
                          max_path_length=MAX_PATH_LENGTH,
                          max_path_width=MAX_PATH_WIDTH)
    if len(config.INPUT_FILES) != 1:
        raise Exception

    data_file = config.INPUT_FILES[0]
    data = pd.read_csv(data_file, names=['path1', 'path2', 'label'])
    features = data.copy()
    labels = features.pop(('label'))

    for i, paths in features.iterrows():
        if i == 0:
            continue

        predict_lines, hash_to_string_dict = path_extractor.extract_paths(paths['path1'])
        raw_prediction_results = model.predict(predict_lines)
        method_prediction_results = common.parse_prediction_results(
            raw_prediction_results, hash_to_string_dict,
            model.vocabs.target_vocab.special_words, topk=SHOW_TOP_CONTEXTS)

        print()
        for raw_prediction, method_prediction in zip(raw_prediction_results, method_prediction_results):
            # print('Original name:\t' + method_prediction.original_name)
            # for attention_obj in method_prediction.attention_paths:
            #     print('%f\tcontext: %s,%s,%s' % (
            #         attention_obj['score'], attention_obj['token1'], attention_obj['path'],
            #         attention_obj['token2']))
            # if self.config.EXPORT_CODE_VECTORS:
            #     print('Code vector:')
            print(' '.join(map(str, raw_prediction.code_vector)))
