import trax
import yaml
import os
import sys
from termcolor import colored


def download_dataset(data_dir):
    """
    This will download the train dataset if no data_dir is specified.
    :param data_dir: the path in which the dataset should be downloaded and stored
    :return: Generators for the training and evaluation set
    """
    # Get generator function for the training set
    train_stream_fn = trax.data.TFDS('opus/medical',
                                     data_dir=data_dir,
                                     keys=('en', 'de'),
                                     eval_holdout_size=0.01,  # 1% for eval
                                     train=True)

    # Get generator function for the eval set
    eval_stream_fn = trax.data.TFDS('opus/medical',
                                    data_dir=data_dir,
                                    keys=('en', 'de'),
                                    eval_holdout_size=0.01,  # 1% for eval
                                    train=False)

    return train_stream_fn, eval_stream_fn


if __name__ == '__main__':
    project_root_directory = sys.path[1]

    with open('../config/model.yml') as config_file:
        model_config = yaml.safe_load(config_file)

    vocabulary_path = model_config.get('vocabulary')['vocabulary_path']
    vocabulary_filename = model_config.get('vocabulary')['vocabulary_filename']
    data_path = os.path.join(project_root_directory, vocabulary_path)

    train_stream_generator, eval_stream_generator = download_dataset(data_path)

    train_stream = train_stream_generator()
    print(colored('train data (en, de) tuple:', 'red'), next(train_stream))
    print()

    eval_stream = eval_stream_generator()
    print(colored('eval data (en, de) tuple:', 'red'), next(eval_stream))


