import argparse
import logging
from tqdm import tqdm

from correctors.pykale_llama_corrector import PykaleLlamaCorrector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_token", help="Huggingface token")
    parser.add_argument("from_index", help="the first broadside to be corrected", default=0)
    parser.add_argument("to_index", help="the last broadside to be corrected", default=1361)
    hf_token = parser.parse_args().hf_token
    from_index = parser.parse_args().from_index
    to_index = parser.parse_args().to_index

    try:
        from_index = int(from_index)
        to_index = int(to_index)
    except ValueError:
        raise ValueError("from_index must be an integer")

    # load broadsides
    logging.info("Loading  eb noisy samples....")
    broadsides_filepath = "/mnt/ceph_rbd/eb_sample_noisy.txt"
    eb_noisy_samples = open(broadsides_filepath).readlines()
    logging.info(f"{len(eb_noisy_samples)} eb samples loaded")
    logging.info(f"creating subset of eb samples from {from_index} to {to_index}")

    # initialise corrector
    logging.info("Initialising corrector.....")
    model_name = "pykale/llama-2-13b-ocr"
    corrector = PykaleLlamaCorrector(model_name, hf_token)

    # correcting text from eb noisy samples
    logging.info("Correcting ocr text in eb noisy samples .......")
    corrected_text = []
    for eb_text in tqdm(eb_noisy_samples):
        corrected_text.append(corrector.correct(eb_text))

    logging.info("eb noisy samples corrected!")

    # save the corrected broadsides to dataframe
    logging.info("save corrected text to")
    result_filepath = f"/mnt/ceph_rbd/eb_sample_corrected_{from_index}_{to_index}.txt"
    logging.info(f"save corrected text to {result_filepath}")
    with open(result_filepath, "w") as f:
        f.write("\n".join(corrected_text))