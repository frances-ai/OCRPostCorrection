import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

class PykaleLlamaCorrector(object):

    def __init__(self, model_name, token):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=token,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True)

    def correct(self, text):
        prompt = f"""### Instruction:
Fix the OCR errors in the provided text.

### Input:
{text}

### Response:
        """
        input_ids = self.tokenizer(prompt, max_length=1024, return_tensors="pt", truncation=True).input_ids.cuda()
        with torch.inference_mode():
            generate_ids = self.model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
        corrected_text = self.tokenizer.batch_decode(generate_ids.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
        return corrected_text







