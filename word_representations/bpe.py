import argparse
import os

import sentencepiece as spm
from tqdm import tqdm


class BPE:

    def __init__(self, input_file, model_path, model_name, training=True):
        self.model = spm.SentencePieceProcessor()
        self.model_path = os.path.join(model_path, model_name)
        self.input_file = input_file

        # when creating the directory don't include the model_name
        # which is already appended to the passed model_path in self.model_path
        if not os.path.exists(model_path):
            print(f'The directory {model_path} does not exist ' +
                  'and will be created')
            os.makedirs(model_path, exist_ok=True)

        if not os.path.exists(self.input_file):
            print(f'The input file does not exist!')

        if not training:
            self._load_model()

    def train_spm(self, vocab_size, char_coverage=1.0):
        if os.path.exists(f'{self.model_path}.model'):
            print(f'The model {self.model_path}.model already exists.')
            return

        print(f'Training model {os.path.basename(self.model_path)}.model')

        assert os.path.exists(self.input_file)

        cmd_args = [
            f'--input={self.input_file}',
            f'--model_prefix={self.model_path}',
            f'--vocab_size={vocab_size}',
            f'--character_coverage={char_coverage}',
            f'--model_type=bpe',
            f'--unk_surface=<unk>'
        ]

        spm.SentencePieceTrainer.Train(' '.join(cmd_args))

    def _load_model(self):
        assert os.path.exists(f'{self.model_path}.model')

        self.model.load(f'{self.model_path}.model')

    def encode_sentence_2_ids(self, sentence):
        return self.model.encode_as_ids(sentence)

    def encode_sentence_2_pieces(self, sentence):
        return self.model.encode_as_pieces(sentence)

    def decode_ids_2_sentence(self, ids):
        return self.model.decode_ids(ids)

    def decode_pieces_2_sentence(self, pieces):
        return self.model.decode_pieces(pieces)

    def encode_text_file_2_ids(self, text_file):
        self._load_model()
        assert os.path.exists(text_file)

        print(f'Using model {os.path.basename(self.model_path)}.model to ' +
              f'encode the text from {text_file} to IDs')

        ids = []
        with open(text_file, 'r', encoding='utf8') as input_file:
            for line in tqdm(input_file.readlines()):
                ids.append(self.encode_sentence_2_ids(line))

        return ids

    def encode_text_file_2_pieces(self, text_file):
        self._load_model()
        assert os.path.exists(text_file)

        print(f'Using model {os.path.basename(self.model_path)}.model to ' +
              f'encode the text from {text_file} to pieces')

        pieces = []
        with open(text_file, 'r', encoding='utf8') as input_file:
            for line in tqdm(input_file.readlines()):
                pieces.append(self.encode_sentence_2_pieces(line))

        return pieces

    def decode_ids_2_text(self, ids):
        self._load_model()
        return [self.decode_ids_2_sentence(entry) for entry in ids]

    def decode_pieces_2_text(self, pieces):
        self._load_model()
        return [self.decode_pieces_2_sentence(entry) for entry in pieces]

    @staticmethod
    def serialise_bpe(output_path, encoded_tokens):
        print(f'Saving results in {output_path}')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf8') as output_file:
            output_list = [' '.join([str(item) for item in entry])
                           for entry in encoded_tokens]
            output_file.write('\n'.join(output_list))

    @staticmethod
    def deserialise_bpe(filename):
        print(f'Reading result from {filename}')
        assert os.path.exists(filename)

        bpe_encoded_data = []
        with open(filename, 'r', encoding='utf8') as file_in:
            for line in file_in.readlines():
                bpe_encoded_data.append(line.strip().split())

        return bpe_encoded_data

    @staticmethod
    def fetch_cli_arguments(parser=None):
        if not parser:
            description = 'BPE word representation script'
            parser = argparse.ArgumentParser(description=description)

        group = parser.add_argument_group('BPE word representation')
        group.add_argument('--input-file-bpe',
                           dest='input_file_bpe',
                           default='',
                           help='The BPE model training dataset')
        group.add_argument('--model-path-bpe',
                           dest='model_path_bpe',
                           help='The base directory in which the model ' +
                                'should be saved')
        group.add_argument('--model-name-bpe',
                           dest='model_name_bpe',
                           help='Filename of the trained model. This will ' +
                                'result in two files: <model_name>.model and ' +
                                '<model_name>.vocab being generated in the ' +
                                '<model_path> directory.')
        group.add_argument('--vocab-size',
                           dest='vocab_size',
                           default=8000,
                           help='The resulting vocabulary size of the BPE ' +
                                'model')
        group.add_argument('--train-bpe',
                           dest='is_train_bpe',
                           action='store_true',
                           default=False,
                           help='Train a BPE model')
        group.add_argument('--encode-bpe',
                           dest='is_encode_bpe',
                           action='store_true',
                           default=False,
                           help='Run a pre-trained BPE model to encode')
        group.add_argument('--decode-bpe',
                           dest='is_decode_bpe',
                           action='store_true',
                           default=False,
                           help='Run a pre-trained BPE model to decode')
        group.add_argument('--ids',
                           dest='is_ids',
                           action='store_true',
                           required=False,
                           help='Signals whether the to encode/decode ' +
                                'to/from IDs or pieces')
        group.add_argument('--encode-file-bpe',
                           dest='encode_file_bpe',
                           required=False,
                           help='The text file whose content needs to be ' +
                                'BPE encoded')
        group.add_argument('--output-file-bpe',
                           dest='output_file_bpe',
                           required=False,
                           help='The file in which the results will be store')

        return parser


def execute_bpe_pipeline(args):
    bpe = BPE(args.input_file_bpe, args.model_path_bpe, args.model_name_bpe)
    if args.is_train_bpe:
        bpe.train_spm(args.vocab_size)

    if args.is_encode_bpe:
        assert args.output_file_bpe

        encode_input_dir = os.path.dirname(args.encode_file_bpe)
        for dataset in ['train', 'dev', 'test']:
            encode_file_bpe = os.path.join(encode_input_dir, f'{dataset}.txt')
            if args.is_ids:
                enc_repr = bpe.encode_text_file_2_ids(encode_file_bpe)
            else:
                enc_repr = bpe.encode_text_file_2_pieces(encode_file_bpe)

            encode_output_dir = os.path.dirname(args.output_file_bpe)
            BPE.serialise_bpe(os.path.join(encode_output_dir, f'{dataset}.txt'),
                              enc_repr)


if __name__ == '__main__':
    main_parser = BPE.fetch_cli_arguments()
    cli_args = main_parser.parse_args()
    execute_bpe_pipeline(cli_args)
