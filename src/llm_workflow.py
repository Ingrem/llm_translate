import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig


class LlmWorkflow:
    """
    Wrapper for quantized HuggingFace causal language models with retryable text generation.
    """

    def __init__(self, default_model=True, model_name=""):
        """
        Initialize workflow with model and tokenizer.

        :param default_model: use default model designed for this workflow
        :param model_name: HuggingFace model name or local path
        """
        if default_model:
            self.model_name = "IlyaGusev/saiga_gemma3_12b"
        else:
            self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm = self._load_model()

    def _load_model(self):
        """Load quantized model in 4-bit precision."""
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        return model

    def _build_inputs(self, prompt: str):
        """Prepare model inputs for generation."""
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)
        return inputs

    def generate_response(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        """
        Generate model response for given prompt.

        :param prompt: Input text
        :param max_tokens: Maximum number of new tokens to generate
        :param temperature: Sampling temperature
        :return: Generated text
        """
        generation_config = GenerationConfig.from_pretrained(
            self.model_name,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        inputs = self._build_inputs(prompt)
        output_ids = self.llm.generate(**inputs, generation_config=generation_config)[0]
        generated_ids = output_ids[len(inputs["input_ids"][0]):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate_with_retry(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        retry_count: int = 10,
        resp_name: str = "llm_response",
    ) -> str:
        """
        Generate response with retry if model returns empty result.

        :param prompt: Input text
        :param max_tokens: Maximum number of new tokens
        :param temperature: Sampling temperature
        :param retry_count: How many times to retry
        :param resp_name: Label for printing
        :return: Model response or fallback message
        """
        for i in range(retry_count):
            response = self.generate_response(prompt, max_tokens=max_tokens, temperature=temperature).split("\n")[0].strip()
            print(f"{resp_name}: {response}")
            if response:
                return response
            prompt = f"Попытка номер {i+1}\n{prompt}"

        return "no response from llm"
