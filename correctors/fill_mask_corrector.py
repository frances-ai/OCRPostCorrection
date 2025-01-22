import pickle
import re

import nltk

from transformers import pipeline

from utils import chunk_text, get_edit_distance, tokenize_text


class FillMaskCorrector(object):
    _default_vocabs_file = "/postcorrection/nckp_ash_vocab_list.pkl"
    def __init__(self,  vocabs_file=_default_vocabs_file, top_k=500):
        # load the vocabs from pickle file
        print("Loading vocabs...")
        with open(vocabs_file, 'rb') as f:
            vocabs = pickle.load(f)
        # make all words in vocabs in lowercase
        vocabs = [vocab.lower() for vocab in vocabs]
        vocabs = list(set(vocabs))
        self.vocabs = vocabs
        # init fill mask mode with top 1000 predictions
        print(f"{len(vocabs)} vocabs loaded!")
        print("Loading model...")
        self.fill_mask = pipeline(
            "fill-mask",
            model="Livingwithmachines/bert_1760_1900",
            top_k=top_k
        )
        print("Model loaded!")

    def fill_mask_for_text(self, text):
        if "[MASK]" not in text:
            return None
        predictions = self.fill_mask(text)
        return predictions

    # define a function to check if a word exists in the vocabs.
    # It first converts word into the base form, then check if the base form, or its lowercase exists in the vocabs.
    # If the word contains multiple hyphen marks, either the whole word or each sides of hyphen mark exist in vocabs, then the word exsits.
    def check_word_exists(self, word):
        # convert the word into its base form
        word = nltk.stem.WordNetLemmatizer().lemmatize(word)
        # print(word)
        exists = False
        if word.lower() in self.vocabs:
            # Hyperpante does not in vocabs, but hyperpante does
            exists = True
        else:
            # check if each sides of hyphen marks exists
            if '-' in word and word[-1] != '-':
                exists = True
                for part in word.split('-'):
                    if part.lower() not in self.vocabs:
                        exists = False
                        break

        return exists

    # define another function to fill one mask with extra conditions:
    # 1. edit distance between masked_word
    # 2. existence in the given vocabs
    def fill_one_mask_for_text(self, text, masked_word):
        # Get top n fill mask predictions, we choose the highest score prediction  which exists in the vocabs,
        # and the edit distance from masked_text and the prediction is no more than defined threshold.
        long_s_marked_word = masked_word.replace('f', 's')
        predictions = self.fill_mask_for_text(text)
        # print(len(predictions))
        filling_word = masked_word
        if predictions is not None:
            for prediction in predictions:
                edit_distance_threshold = len(masked_word) / 2
                if edit_distance_threshold > 3:
                    edit_distance_threshold = 3
                if (get_edit_distance(masked_word,
                                      prediction['token_str']) <= edit_distance_threshold or get_edit_distance(
                        long_s_marked_word, prediction['token_str']) <= edit_distance_threshold) and self.check_word_exists(
                        prediction['token_str']):
                    # print(prediction['token_str'])
                    # as the prediction are all lower case, we need to convert it back to captilizaed one if the
                    # first letter of masked_word is the same as the one in the prediction
                    filling_word = prediction['token_str']
                    if masked_word[0].isupper() and masked_word[0].lower() == filling_word[0]:
                        filling_word = prediction['token_str'].capitalize()
                    break
                else:
                    # print(prediction['token_str'])
                    pass

        return filling_word

    def mask_text(self, text, from_index):
        """
        create a masked_text by replacing the first word (staring from a given index) does not exists in vocabs with a mask,
        also get the masked word.
        """
        tokens = tokenize_text(text)
        return self.mask_text_from_tokens(tokens, from_index)

    def mask_text_from_tokens(self, tokens, from_index):
        """
        create a masked_text by replacing the first word (staring from a given index) does not exist in vocabs with a mask,
        also get the masked word. This function will utilise the tokens from the original text to optimize the performance.
        """
        new_tokens = tokens.copy()
        masked_word = None
        masked_index = -1
        potential_errors = ["ot", "ol", "o", "f", "Raman", "capfule", "capfules", "bv", "pn", "tne", "fiom", "iu", "hut",
                           "md", "verv"]
        for index in range(from_index, len(new_tokens)):
            token = tokens[index]
            if token in potential_errors or not self.check_word_exists(token):
                # maybe add more filterring options to reduce the number of masks
                # TODO: mask for words with hyphen mark
                new_tokens[index] = ('[MASK]')
                masked_word = token
                masked_index = index
                break
        masked_text = ' '.join(new_tokens)
        return masked_text, masked_word, masked_index

    # Now, let's put everything together to create a single function for correct OCR errors in a text
    def correct_short_text(self, text):
        # 1. create a masked_text by replacing one mask for the first word does not exists in vocabs, also get the masked word
        # 2. correct the masked_word by filling the mask for the masked_text
        # 3. repeat this step until there is no mask can be replaced in the corrected masked text
        from_index = 0
        tokens = tokenize_text(text)
        masked_text, masked_word, masked_index = self.mask_text_from_tokens(tokens, from_index)
        while masked_word is not None:
            candidate_word = self.fill_one_mask_for_text(masked_text, masked_word)
            if candidate_word == masked_word or get_edit_distance(candidate_word, masked_word) > 2:
                # try small correction methods
                # one problem with this is that the priority
                # 1. words which can corrected by removing non-English-letters
                english_letters_masked_word = re.sub(r'[^a-zA-Z]', '', masked_word)
                fs_masked_word = masked_word.replace('f', 's')
                if self.check_word_exists(english_letters_masked_word):
                    candidate_word = english_letters_masked_word
                elif self.check_word_exists(fs_masked_word):
                    candidate_word = fs_masked_word
            tokens[masked_index] = candidate_word
            from_index = masked_index + 1
            masked_text, masked_word, masked_index = self.mask_text_from_tokens(tokens, from_index)
        return masked_text

    # Now, let's put everything together to create a single function for correct OCR errors in a text
    def correct(self, text):
        # chunk the text if it's too long
        texts = chunk_text(text, max_tokens=30)

        corrected_chunks = []
        for text in texts:
            corrected_chunk = self.correct_short_text(text)
            corrected_chunks.append(corrected_chunk)

        return ' '.join(corrected_chunks)


