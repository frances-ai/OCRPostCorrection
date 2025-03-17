import argparse
import logging
from tqdm import tqdm

import pandas as pd

from correctors.pykale_llama_corrector import PykaleLlamaCorrector

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    # load sequences from dataset volume
    logging.info("Loading sequences ....")
    sequences_filepath = "/mnt/ceph_rbd/test.csv"
    sequences_df = pd.read_csv(sequences_filepath)
    logging.info(f"{len(sequences_df)} sequences loaded")

    # group sequences into articles
    # Keep only sample ID, OCR Text and Ground Truth, group texts with same Sample ID
    logging.info("Grouping sequences into articles....")
    sequences_simple_df = sequences_df[["Sample ID", "OCR Text", "Ground Truth"]]
    articles_df = sequences_simple_df.groupby("Sample ID").agg({"OCR Text": " ".join, "Ground Truth": " ".join}).reset_index()
    logging.info(f"{len(articles_df)} articles grouped")
    articles_ocr_text = articles_df["OCR Text"].tolist()

    # initialise corrector
    logging.info("Initialising corrector.....")
    model_name = "pykale/llama-2-13b-ocr"
    corrector = PykaleLlamaCorrector(model_name)

    # correcting text from articles_ocr_text
    logging.info("Correcting ocr text.......")
    corrected_texts = []
    for ocr_text in tqdm(articles_ocr_text):
        corrected_text = corrector.correct(ocr_text)
        corrected_texts.append(corrected_text)

    logging.info(f"{len(corrected_texts)} ocr text corrected")

    # save the corrected text to csv file
    articles_df["Corrected Text"] = corrected_texts
    result_filepath = "/mnt/ceph_rbd/articles_result.csv"
    articles_df.to_csv(result_filepath, index=False)
