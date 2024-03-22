from cog import BasePredictor, Input
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

MODEL_NAME = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
MODEL_CACHE = "cache"

DEFAULT_PROMPT = "Tell me about AI"


class Predictor(BasePredictor):
    def setup(self):
        print("Starting setup")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True, cache_dir=MODEL_CACHE
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", trust_remote_code=True, revision="main"
        )

    def predict(
        self,
        prompt: str = Input(
            description="Prompt to send to the model.", default=DEFAULT_PROMPT
        ),
        # system_prompt: str = Input(
        #     description="System prompt that helps guide system behavior",
        #     default="You are a helpful, respectful and honest assistant",
        # ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=128,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.7,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=0,
            default=40,
        ),
        repetition_penalty: float = Input(
            description="A parameter that controls how repetitive text can be. Lower means more repetitive, while higher means less repetitive. Set to 1.0 to disable.",
            ge=0.0,
            default=1.15,
        ),
    ) -> str:

        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids().cuda()
        # outputs = self.model.generate(
        #     inputs=input_ids,
        #     temperature=temperature,
        #     top_k=top_k,
        #     top_p=top_p,
        #     repetition_penalty=repetition_penalty,
        #     max_new_tokens=max_new_tokens,
        # )

        # output = self.tokenizer.decode(outputs[0])

        # print(output)

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        output = pipe(prompt)[0]["generated_text"]
        final = output.split("ASSISTANT:")
        return final[-1]
