import argparse
import os.path

import torch.cuda

import datasets
import models


class BaseConfig:

    def __init__(self):
        self.name = 'base options'

        NETWORKS = ['BERT']
        DATASETS = ['wikitext2']

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--model', type=str, default='BERTLM', help='Choose the network to work with.',
                            choices=NETWORKS)
        parser.add_argument('--dataset', type=str, default='wikitext2', help='Choose the dataset to train on.',
                            choices=DATASETS)

        parser.add_argument('--vocab_path', type=str, default="data/wikitext2.pkl",
                            help='built vocab model path with bert-vocab')
        parser.add_argument('--vocab_max_size', type=int, default=None, help='Number of epochs to train for.')
        parser.add_argument('--vocab_min_frequency', type=int, default=1,
                            help='The minimum frequency needed to include a token in the vocabulary. Values less than'
                                 '1 will be set to 1. Default: 1')

        parser.add_argument('--storage_directory', type=str, default=os.path.dirname(os.path.abspath(__file__)).replace(
            '\\config', '').replace('/config', ''),
                            help='Directory where all data generated during execution will be stored')

        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help=
                            'Execution device, cuda is default if device has a cuda ready GPU', choices=['cpu', 'cuda'])

        parser.add_argument('--print_samples', dest='print_samples', action='store_true')
        parser.add_argument('--print_no_samples', dest='print_samples', action='store_false')
        parser.set_defaults(dataset_depth_from_norm=True)

        parser.add_argument("-o", "--output_path",
                            default="output/", type=str,
                            help="ex)output/bert.model")

        parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")

        parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")


        self.parser = parser
        self.parser = self.gather()

    def get_parser(self):
        return self.parser

    def get(self):
        return self.parser.parse_args()

    def gather(self):
        options, _ = self.parser.parse_known_args()
        model = models.get(options.model)
        self.parser = model.extend_parser(self.parser)

        dataset = datasets.get(options.dataset)
        self.parser = dataset.extend_parser(self.parser)

        return self.parser

    def print(self):
        config = self.get()

        message = '-' * 84 + '\n'
        message += '|{o:<30}|{v:<30}|{d:<20}|\n'.format(o=self.name, v='value', d='default')
        message += '-' * 84 + '\n'
        for name, value in sorted(vars(config).items()):
            name = str(name)
            value = str(value)
            comment = ''
            default = str(self.parser.get_default(name))
            if value != default:
                comment = default
                if len(comment) > 20:
                    comment = comment[0:18] + '..'
            if len(name) > 30:
                name = name[0:28] + '..'
            if len(value) > 30:
                value = value[0:28] + '..'
            message += '|{n:<30}|{v:<30}|{c:<20}|\n'.format(n=name, v=value, c=comment)
        message += '-' * 84 + '\n'
        message += '|End {o:<78}|\n'.format(o=self.name)
        message += '-' * 84 + '\n'
        print(message)
