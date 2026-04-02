# Simple wrapper around llama-server
import atexit
import subprocess
from typing import Iterable

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class Server:
    def __init__(self, bin_path: str, model_path: str, startup_args: list[str]):
        self.bin_path = bin_path
        self.model_path = model_path
        self.startup_args = startup_args
        self.log = open("server.log", "w")
        self.process = None

    def open(self):
        if self.process:
            atexit.unregister(self.process.terminate)
            self.process.terminate()

        self.process = subprocess.Popen(
            [self.bin_path, '-m', self.model_path] + self.startup_args,
            stdout=self.log,
            stderr=self.log,
        )
        self.client = OpenAI(api_key="no_money_for_sam", base_url="http://localhost:8080/v1")
        atexit.register(self.process.terminate)

    def close(self):
        if self.process:
            self.process.terminate()
            atexit.unregister(self.process.terminate)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def complete(self, messages: Iterable[ChatCompletionMessageParam]) -> str|None:
        return self.client.chat.completions.create(
            messages=messages,
            model="model",
        ).choices[0].message.content
