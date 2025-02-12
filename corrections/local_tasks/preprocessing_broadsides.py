import logging
from tqdm import tqdm
import spacy

import pandas as pd

nlp = spacy.load("en_core_web_sm")

def text_to_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # if sentence has less than 3 words, then it will be added to previous sentence
    s_size = len(sentences)
    s_index = 0
    while s_index < s_size:
        if len(sentences[s_index].split()) < 3:
            if s_index > 0:
                sentences[s_index - 1] += " " + sentences[s_index]
                sentences.pop(s_index)
                s_size -= 1
        s_index += 1

    # if sentence ends with dash or em dash, then the next sentence will be added
    s_size = len(sentences)
    s_index = 0
    while s_index < s_size:
        if sentences[s_index].endswith('—') or sentences[s_index].endswith('—'):
            if s_index + 1 < s_size:
                sentences[s_index] += " " + sentences[s_index + 1]
                sentences.pop(s_index + 1)
                s_size -= 1
        s_index += 1
    return sentences

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # load broadsides
    logging.info("Loading broadsides dataframe ....")
    broadsides_filepath = "/Users/lilinyu/Documents/PhD/InformationExtraction/Broadsides/NLS/broadsides_dataframe"
    broadsides_df = pd.read_json(broadsides_filepath, orient='index')
    logging.info(f"{len(broadsides_df)} broadsides loaded")

    # create a list broadsides with volume id, text and ground truth
    logging.info("Create a list broadsides with volume id, text and ground truth ....")
    broadsides = []

    # add broadsides with more than one page, merge the text for these broadsides
    bs_vid_more_than_one_page = broadsides_df[broadsides_df['numberOfPages'] > 1]["volumeId"].unique()
    for volume_id in bs_vid_more_than_one_page:
        current_bs_df = broadsides_df[broadsides_df['volumeId'] == volume_id]
        current_bs = {
            'volumeId': volume_id,
            'text': "",
            'ground truth': ""
        }
        for index, row in current_bs_df.iterrows():
            current_bs['text'] += " " + row['text']
            current_bs['ground truth'] = row['ground truth']
        broadsides.append(current_bs)

    # add the rest broadsides
    for index, row in broadsides_df[broadsides_df['numberOfPages'] == 1].iterrows():
        current_bs = {
            'volumeId': row['volumeId'],
            'text': row['text'],
            'ground truth': row['ground truth']
        }
        broadsides.append(current_bs)

    # chunk text and ground truth into sentences for each broadside using spacy
    logging.info("chunk text and ground truth into sentences for each broadside using spacy ....")
    for bs in tqdm(broadsides):
        bs['sentences'] = text_to_sentences(bs['text'])

    # save the broadsides to dataframe
    result_df = pd.DataFrame(broadsides)
    result_filepath = "broadsides_sentences_df.json"
    result_df.to_json(result_filepath, orient='index')