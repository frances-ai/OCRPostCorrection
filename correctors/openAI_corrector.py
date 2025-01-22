from openai import OpenAI

from utils import chunk_text

API_KEY = "your api key"

class OpenAICorrector(object):
    _default_model_name = 'gpt-4o'
    def __init__(self, model_name=_default_model_name):
        self.client = OpenAI(api_key=API_KEY)
        self.model_name = model_name
        self.system_prompt = "You are an assistant trained to correct text from OCR outputs that may contain errors. Your task is to reconstruct the likely original text. Restore the text to its original form, including handling non-standard elements that aligns with their intended meaning and use."
        self.user_prompt = """
        ###Instruction###

        Reconstruct the likely original text based on the OCR output provided. Interpret the possible errors introduced by the OCR process and correct them to best represent the initial text. Only provide the corrected version, do not say any other words in your response. You will be penalized for adding extra words.
        
        ###OCR text###
        
        %s
        """

    def correct_short_text(self, text):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt % text,
                }
            ]
        )
        return completion.choices[0].message.content

    def correct(self, text):
        # chunk the text if it's too long
        texts = chunk_text(text, max_tokens=30)

        corrected_chunks = []
        for text in texts:
            corrected_chunk = self.correct_short_text(text)
            corrected_chunks.append(corrected_chunk)

        return ' '.join(corrected_chunks)
