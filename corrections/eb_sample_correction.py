import argparse
import logging
from tqdm import tqdm
import pandas as pd

from correctors.llama_corrector import LlamaCorrector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="name or path of fine-tuned model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("from_index", help="the first eb article to be corrected", default=0)
    parser.add_argument("to_index", help="the last eb article to be corrected", default=1361)
    model_name = parser.parse_args().model_name
    from_index = parser.parse_args().from_index
    to_index = parser.parse_args().to_index

    try:
        from_index = int(from_index)
        to_index = int(to_index)
    except ValueError:
        raise ValueError("from_index must be an integer")

    # load broadsides
    logging.info("Loading  eb noisy samples....")
    eb_filepath = "/mnt/ceph_rbd/eb_samples.json"
    eb_samples_df = pd.read_json(eb_filepath, orient="records", lines=True)
    logging.info(f"{len(eb_samples_df)} eb samples loaded")
    logging.info(f"creating subset of eb samples from {from_index} to {to_index}")
    eb_samples_df = eb_samples_df.iloc[from_index:to_index]
    eb_noisy_samples = eb_samples_df['ocr text'].tolist()

    # initialise corrector
    logging.info(f"Initialising corrector with model: {model_name}.....")
    corrector = LlamaCorrector(model_name)

    # correcting text from eb noisy samples
    logging.info("Correcting ocr text in eb noisy samples .......")
    corrected_text = []
    for eb_text in tqdm(eb_noisy_samples):
        corrected_text.append(corrector.correct(eb_text))

    logging.info("eb noisy samples corrected!")

    # save the corrected eb to dataframe
    eb_samples_df['corrected'] = corrected_text
    result_filepath = f"/mnt/ceph_rbd/eb_samples_corrected_{from_index}_{to_index}.json"
    logging.info(f"save dataframe with corrected text to {result_filepath}")
    eb_samples_df.to_json(result_filepath, orient="records", lines=True)
    logging.info('Done!')
