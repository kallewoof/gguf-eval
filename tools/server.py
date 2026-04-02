# Simple wrapper around llama-server
import atexit
import json
import subprocess
import time
from typing import Iterable

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


echo_fns = {
    True: lambda token: print(token, end="", flush=True),
    False: lambda _: None
}

class Server:
    def __init__(self, bin_path: str, model_path: str, startup_args: list[str]):
        self.bin_path = bin_path
        self.model_path = model_path
        self.startup_args = startup_args # + ["--verbose"]
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
        atexit.register(self.process.terminate)

        self.client = OpenAI(api_key="no_money_for_sam", base_url="http://localhost:8080/v1")
        while True:
            print("[sleeping]")
            time.sleep(1)
            print("[slept]")
            try:
                print("[calling chat.completions.create]")
                self.client.chat.completions.create(
                    messages=[{"role":"user","content":"hi"}],
                    model="model",
                    max_tokens=1,
                    stream=True,
                )
                print("[called, no exception]")
            except Exception as e:
                if "Loading model" in f"{e}":
                    print(".", end="", flush=True)
                    continue
                print(f"{e}")
                raise
            print("[breaking]")
            break

    def close(self):
        if self.process:
            self.process.terminate()
            atexit.unregister(self.process.terminate)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def complete(self, messages: Iterable[ChatCompletionMessageParam], echo=False, max_tokens=4096, **kwargs) -> dict[str,str]:
        p = echo_fns[echo]
        response_dict: dict[str,list[str]] = {}

        state = "unknown"
        def S(s,t):
            nonlocal state
            if state != s:
                state = s
                response_dict[state] = []
                if echo:
                    print(f"\n<{state}>\n")
            p(t)
            response_dict[state].append(t)

        with self.client.chat.completions.with_streaming_response.create(
            model="model",
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if not (line or '').startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if token := delta.get("content"):
                    S("response", token)
                elif token := delta.get("reasoning_content"):
                    S("reasoning", token)

        return {
            "reasoning": ''.join(response_dict['reasoning']),
            "response": ''.join(response_dict['response']),
        }
