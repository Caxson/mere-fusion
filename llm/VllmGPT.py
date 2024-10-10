import json
import requests
# from core import content_db

class VllmGPT:

    def __init__(self, host="192.168.1.3",
                 port="8101",
                 model="THUDM/chatglm3-6b",
                 max_tokens="1024"):
        self.host = host
        self.port = port
        self.model=model
        self.max_tokens=max_tokens
        self.__URL = "http://{}:{}/v1/completions".format(self.host, self.port)
        self.__URL2 = "http://{}:{}/v1/chat/completions".format(self.host, self.port)

    def chat(self,cont):
        chat_list = []
        content = {
            "model": self.model,
            "prompt":"Simple reply;" +  cont,
            "history":chat_list}
        url = self.__URL
        req = json.dumps(content)
        
        headers = {'content-type': 'application/json'}
        r = requests.post(url, headers=headers, data=req)
        res = json.loads(r.text)
        
        return res['choices'][0]['text']

    def question2(self,cont):
        chat_list = []
        content = {
            "model": self.model,
            "prompt":"Simple reply;" +  cont,
            "history":chat_list}
        url = self.__URL2
        req = json.dumps(content)
        
        headers = {'content-type': 'application/json'}
        r = requests.post(url, headers=headers, data=req)
        res = json.loads(r.text)
        
        return res['choices'][0]['message']['content']
    
if __name__ == "__main__":
    vllm = VllmGPT('192.168.1.3','8101')
    req = vllm.chat("hello?")
    print(req)
