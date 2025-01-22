import argparse
import logging
from tqdm import tqdm

import pandas as pd

from correctors.pykale_llama_corrector import PykaleLlamaCorrector

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_token", help="Huggingface token")
    hf_token = parser.parse_args().hf_token

    # load sequences from dataset volume
    logging.info("Loading sequences ....")
    sequences_filepath = "/mnt/ceph_rbd/test.csv"
    sequences_df = pd.read_csv(sequences_filepath)
    logging.info(f"{len(sequences_df)} sequences loaded")
    sequences_ocr_text = sequences_df["OCR Text"].tolist()

    # initialise corrector
    logging.info("Initialising corrector.....")
    model_name = "pykale/llama-2-13b-ocr"
    corrector = PykaleLlamaCorrector(model_name, hf_token)

    # correcting text from sequences_ocr_text
    logging.info("Correcting ocr text.......")
    corrected_texts = []
    for ocr_text in tqdm(sequences_ocr_text):
        corrected_text = corrector.correct(ocr_text)
        corrected_texts.append(corrected_text)

    logging.info(f"{len(corrected_texts)} ocr text corrected")

    # save the corrected text to csv file
    sequences_df["Corrected Text"] = corrected_texts
    result_filepath = "/mnt/ceph_rbd/result.csv"
    sequences_df.to_csv(result_filepath, index=False)
