import argparse
import logging
from tqdm import tqdm

import pandas as pd

from correctors.pykale_llama_corrector import PykaleLlamaCorrector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_token", help="Huggingface token")
    parser.add_argument("from_index", help="the first broadside to be corrected", default=0)
    parser.add_argument("to_index", help="the last broadside to be corrected", default=1726)
    hf_token = parser.parse_args().hf_token
    from_index = parser.parse_args().from_index
    to_index = parser.parse_args().to_index

    try:
        from_index = int(from_index)
        to_index = int(to_index)
    except ValueError:
        raise ValueError("from_index must be an integer")

    # load broadsides
    logging.info("Loading broadsides dataframe ....")
    broadsides_filepath = "/mnt/ceph_rbd/broadsides_sentences_df.json"
    broadsides_df = pd.read_json(broadsides_filepath, orient='index')
    logging.info(f"{len(broadsides_df)} broadsides loaded")
    logging.info(f"creating subset of broadsides from {from_index} to {to_index}")
    broadsides_df = broadsides_df.iloc[from_index:to_index]
    logging.info(f"{len(broadsides_df)} broadsides will be corrected")
    broadsides_sentences = broadsides_df["sentences"].tolist()

    # initialise corrector
    logging.info("Initialising corrector.....")
    model_name = "pykale/llama-2-13b-ocr"
    corrector = PykaleLlamaCorrector(model_name, hf_token)

    # correcting text from sequences_ocr_text
    logging.info("Correcting ocr text in broadsides .......")
    corrected_sents_count = 0
    total_corrected_sentences = []
    for bs in tqdm(broadsides_sentences):
        corrected_sentences = []
        for sentence in bs:
            corrected_sentence = corrector.correct(sentence)
            corrected_sentences.append(corrected_sentence)

        total_corrected_sentences.append(corrected_sentences)
        corrected_sents_count += len(corrected_sentences)

    logging.info(f"{len(broadsides_sentences)} broadsides, with {corrected_sents_count} sentences corrected")

    # save the corrected broadsides to dataframe
    broadsides_df["Corrected Sentences"] = total_corrected_sentences
    result_filepath = "/mnt/ceph_rbd/broadsides_results_df.json"
    broadsides_df.to_json(result_filepath, orient='index')