import argparse
import logging
from tqdm import tqdm
import pandas as pd

from correctors.llama_corrector import LlamaCorrector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="name or path of fine-tuned model", default="pykale/llama-2-13b-ocr")
    parser.add_argument("from_index", help="the first eb article to be corrected", default=2250)
    parser.add_argument("to_index", help="the last eb article to be corrected", default=13459)
    model_name = parser.parse_args().model_name
    from_index = parser.parse_args().from_index
    to_index = parser.parse_args().to_index

    try:
        from_index = int(from_index)
        to_index = int(to_index)
    except ValueError:
        raise ValueError("from_index must be an integer")

    # load broadsides
    logging.info("Loading  eb articles....")
    eb_filepath = "/mnt/ceph_rbd/eb_articles.json"
    eb_articles_df = pd.read_json(eb_filepath, orient="records", lines=True)
    logging.info(f"{len(eb_articles_df)} eb articles loaded")
    logging.info(f"creating subset of eb samples from {from_index} to {to_index}")
    eb_articles_df = eb_articles_df.iloc[from_index:to_index]
    eb_lq_descriptions = eb_articles_df['lq_description'].tolist()
    eb_desc_chunks_offsets = eb_articles_df['chunk_offsets'].tolist()

    # initialise corrector
    logging.info(f"Initialising corrector with model: {model_name}.....")
    corrector = LlamaCorrector(model_name)

    # correcting text from eb noisy samples
    logging.info("Correcting ocr text in eb low quality descriptions .......")
    total_corrected_chunks = []
    for eb_text, offsets in zip(tqdm(eb_lq_descriptions), eb_desc_chunks_offsets):
        corrected_chunks = []
        for offset in offsets:
            corrected_chunk = corrector.correct(eb_text[offset['start']:offset['end']])
            corrected_chunks.append(corrected_chunk)
        total_corrected_chunks.append(corrected_chunks)

    logging.info("eb low quality descriptions  corrected!")

    # save the corrected eb to dataframe
    eb_articles_df['corrected_chunks'] = total_corrected_chunks
    result_filepath = f"/mnt/ceph_rbd/eb_corrected_{from_index}_{to_index}.json"
    logging.info(f"save dataframe with corrected text to {result_filepath}")
    eb_articles_df.to_json(result_filepath, orient="records", lines=True)
    logging.info('Done!')
