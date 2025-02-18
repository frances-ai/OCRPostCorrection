import argparse
import logging
from tqdm import tqdm

import pandas as pd

from correctors.openAI_corrector import OpenAICorrector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="open ai model name")
    model_name = parser.parse_args().model_name

    # load broadsides
    logging.info("Loading broadsides dataframe ....")
    broadsides_filepath = "broadsides_sentences_df.json"
    broadsides_df = pd.read_json(broadsides_filepath, orient='index')
    logging.info(f"{len(broadsides_df)} broadsides loaded")
    broadsides_sentences = broadsides_df["sentences"].tolist()

    # initialise corrector
    logging.info("Initialising corrector.....")
    try:
        corrector = OpenAICorrector(model_name=model_name)
    except Exception as e:
        logging.error(f"Could not initialise corrector for {model_name}: {e}")
        exit(1)

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
    result_filepath = f"broadsides_results_df_{model_name}.json"
    broadsides_df.to_json(result_filepath, orient='index')