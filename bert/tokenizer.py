from transformers import BertTokenizer
from tqdm import tqdm
import torch


class Tokenizer:
    def __init__(self, pretrained_model_name, lower_case):
        self.model = None
        self.pretrained_model_name = pretrained_model_name
        self.lower_case = lower_case

    def load(self, model_dir):
        self.model = BertTokenizer.from_pretrained(model_dir)

    def save(self, model_dir):
        self.model.save_pretrained(model_dir)

    def tokenize(self, sentences, sentence_length):
        if self.model is None:
            print('Loading BERT tokenizer...')
            self.model = BertTokenizer.from_pretrained(
                                                self.pretrained_model_name,
                                                do_lower_case=self.lower_case)

        # Tokenize all of the sentences and map the tokens to their word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        print("Encoding sentences...")
        for sent in tqdm(sentences):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.model.encode_plus(sent,
                                                  add_special_tokens = True,
                                                  max_length = sentence_length,
                                                  pad_to_max_length = True,
                                                  return_attention_mask = True,
                                                  return_tensors = 'pt',
                                                  truncation=True)

            input_ids.append(encoded_dict['input_ids'])

            # The attention mask simply differentiates padding from non-padding)
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks


    def get_max_sentence_length(self, sentences):
        """Get the max encoded sentence length from the given sentences.
        This allows to choose the right sentence length when tokenizing the
        sentences (the model will truncate or pad any smaller or bigger
        sentence)
        """
        # Find the max sentence length
        if self.model is None:
            self.model = BertTokenizer.from_pretrained(
                                                self.pretrained_model_name,
                                                do_lower_case=self.lower_case)
        max_len = 0
        for sent in sentences:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.model.encode(sent, add_special_tokens=True)
            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
        print('Max sentence length (encoded): ', max_len)
        return max_len
