import argparse
import logging
from tqdm import tqdm

from correctors.openAI_corrector import OpenAICorrector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="open ai model name")
    model_name = parser.parse_args().model_name

    # load broadsides
    logging.info("Loading eb noisy samples....")
    eb_sample_noisy_filepath = "eb_sample_noisy.txt"
    # read noisy samples into list
    noisy_samples = open(eb_sample_noisy_filepath).readlines()
    noisy_samples = noisy_samples[:100]
    logging.info(f"{len(noisy_samples)} samples loaded")

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
    for ocr_text in tqdm(noisy_samples):
        corrected_text = corrector.correct(ocr_text)
        corrected_texts.append(corrected_text)

    logging.info(f"{len(corrected_texts)} ocr text corrected")

    # save the corrected text to csv file
    result_filepath = "eb_sample_corrected.txt"
    with open(result_filepath, "w") as f:
        f.write("\n".join(corrected_texts))