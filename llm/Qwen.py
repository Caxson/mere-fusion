import os
import openai

from modelscope import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Qwen:
    def __init__(self, model_path="Qwen/Qwen-1_8B-Chat", api_base=None, api_key=None) -> None:
        self.local = True
        if api_key is not None and api_base is not None:
            openai.api_base = api_base
            openai.api_key = api_key
            self.local = False
            return

        self.model, self.tokenizer = self.init_model(model_path)
        self.data = {}

    def init_model(self, path="Qwen/Qwen-1_8B-Chat"):
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat",
                                                     device_map="auto",
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        return model, tokenizer

    def chat(self, question):
        if not self.local:
            response = openai.ChatCompletion.create(
                model="Qwen",
                messages=[
                    {"role": "user", "content": question}
                ],
                stream=False,
                stop=[]
            )
            return response.choices[0].message.content
        self.data["question"] = f"{question} ### Instruction:{question}  ### Response:"
        try:
            response, history = self.model.chat(self.tokenizer, self.data["question"], history=None)
            print(history)
            return response
        except:
            return "Sorry, your request has encountered an error. Please try again.\n"


def test():
    llm = Qwen(model_path="Qwen/Qwen-1_8B-Chat")
    answer = llm.chat(question="hello？")
    print(answer)


if __name__ == '__main__':
    test()
