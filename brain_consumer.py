import logging
import threading
import queue
import time
from openai_server import *

# 定义一个全局队列
text_queue = queue.Queue()
video_queue = queue.Queue()

logging.basicConfig(level=logging.INFO)


# 生产者函数，用于随时添加数据到队列
def produce(text):
    logging.info(f"Producer is producing: {text}")
    text_queue.put(text)


def video_produce(text):
    logging.info(f"Video producer is producing: {text}")
    video_queue.put(text)


# 消费者函数，持续监控队列并消费数据
def consumer():
    logging.info(f"Consumer_Log ---------------------------------- consuming :{inputting}，{text_queue.empty()}")
    if inputting is False and not text_queue.empty():
        # video = video_queue.get()
        video = "person 1, age:30, man, happy, asia"
        if not video_queue.empty():
            video = video_queue.get()
        text = text_queue.get()  # 从队列中获取数据
        logging.info(f"Consumer_Log ---------------------------------- consuming text: {text}, video: {video}")
        # messages = [
        #     {'role': 'user',
        #      'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
        # ]
        input_messages = [
            {"role": "system",
             "content": "你需要扮演一个人类，现在需要不断根据我发给你的画面数据和聊天数据生成对应的回答和响应，你需要把我发给你的文本当作我给你说的话，我发给你的画面当作你看到的，你只需要按照文本内容给出响应，画面内容只作为参考,用中文回复并把自己当作人来应答，确保您的输出格式可以直接从您的响应中大声读出(它将被转换成口语)。"},
            {"role": "user", "content": f"文本（前两个数字是文本开始和结束长度）：{text}\n -画面：{video}"},
        ]
        stream_chat(input_messages)
    # 在 1 秒后再次调用 consumer 函数
    logging.info(f"Consumer_Log ---------------------------------- consuming :start")
    threading.Timer(1, consumer).start()


# 初始化消费者线程
def start_consumer_thread():
    threading.Timer(1, consumer).start()
