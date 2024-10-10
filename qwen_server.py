import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)

# device = "cuda"  # the device to load the model onto
device = "mps"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# Now you do not need to add "trust_remote_code=True"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map={"": device},
).to(device)

inputting = False

# 普通消息
def chat(messages):
    inputting = True
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Directly use generate() and tokenizer.decode() to get the output.
    # Use `max_new_tokens` to control the maximum output length.
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    logging.info(f"QWen-Response: {response}")
    inputting = False
    return response

# 管道消息
def pipeline_chat(messages):
    inputting = True
    from transformers import pipeline

    pipe = pipeline("text-generation", "Qwen/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto")

    response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]
    logging.info(f"QWen-Response: {response_message}")
    inputting = False
    return response_message

# 批量管道消息
def batch_pipeline_chat(message_batch):
    inputting = True
    from transformers import pipeline

    pipe = pipeline("text-generation", "Qwen/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto")
    pipe.tokenizer.padding_side = "left"

    if message_batch is None:
        message_batch = [
            [{"role": "user", "content": "Give me a detailed introduction to large language model."}],
            [{"role": "user", "content": "Hello!"}],
        ]

    result_batch = pipe(message_batch, max_new_tokens=512, batch_size=2)
    response_message_batch = [result[0]["generated_text"][-1] for result in result_batch]
    logging.info(f"QWen-Response: {response_message_batch}")
    inputting = False
    return response_message_batch

def stream_queue_chat(messages):
    inputting = True
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Starting here, we add streamer for text generation.
    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Use Thread to run generation in background
    # Otherwise, the process is blocked until generation is complete
    # and no streaming effect can be observed.
    from threading import Thread
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end='')
    inputting = False
    print("\nFinal generated text:", generated_text)

from transformers import TextStreamer

# 自定义 TextStreamer 类
class CollectingTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.collected_text = ""  # 用于收集文本的缓冲区

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.collected_text == "":
            logging.info(f"first output text time: {time.time()}")
        """打印并收集生成的文本。"""
        self.collected_text += text  # 收集文本
        print(text, flush=True, end="" if not stream_end else "\n")  # 实时打印新增的文本

def stream_chat(messages):
    inputting = True
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    streamer = CollectingTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 生成文本并实时打印到控制台
    model.generate(
        **model_inputs,
        max_new_tokens=512,
        streamer=streamer,
    )

    # 生成完成后，收集到的完整文本
    collected_text = streamer.collected_text
    inputting = False
    logging.info("\nCollected_Text:", collected_text)

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
# prompt = "Give me a short introduction to large language model."
# messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt},
#     ]

# start_time = time.time()
# response = chat(messages)
# end_time = time.time()
# logging.info(f"{chat.__name__} took {(end_time - start_time) * 1000} milliseconds to execute.")

# start_time = time.time()
# print(f"QWen-Server Started: {start_time}")
# stream_chat(messages)
# end_time = time.time()
# logging.info(f"{stream_chat.__name__} took {(end_time - start_time) * 1000} milliseconds to execute.")

# start_time = time.time()
# pipeline_chat(messages)
# end_time = time.time()
# logging.info(f"{pipeline_chat.__name__} took {(end_time - start_time) * 1000} milliseconds to execute.")
#
# start_time = time.time()
# batch_pipeline_chat(None)
# end_time = time.time()
# logging.info(f"{batch_pipeline_chat.__name__} took {(end_time - start_time) * 1000} milliseconds to execute.")

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt},
# ]
#
# messages.append({"role": "assistant", "content": response})
#
# prompt = "Tell me more."
# messages.append({"role": "user", "content": prompt})
#
# chat(messages)
