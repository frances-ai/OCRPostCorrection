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
    logging.info("Loading bln600 sequences....")
    sequences_filepath = "bln600_sequences.csv"
    sequences_df = pd.read_csv(sequences_filepath)
    logging.info(f"{len(sequences_df)} sequences loaded")
    sequences_ocr_text = sequences_df["OCR Text"].tolist()

    # initialise corrector
    logging.info("Initialising corrector.....")
    try:
        corrector = OpenAICorrector(model_name=model_name)
    except Exception as e:
        logging.error(f"Could not initialise corrector for {model_name}: {e}")
        exit(1)

    # correcting text from sequences_ocr_text
    logging.info("Correcting ocr text.......")
    corrected_texts = []
    for ocr_text in tqdm(sequences_ocr_text):
        corrected_text = corrector.correct(ocr_text)
        corrected_texts.append(corrected_text)

    logging.info(f"{len(corrected_texts)} ocr text corrected")

    # save the corrected text to csv file
    sequences_df["Corrected Text"] = corrected_texts
    result_filepath = f"bln600_sequences_result_{model_name}.csv"
    sequences_df.to_csv(result_filepath, index=False)
