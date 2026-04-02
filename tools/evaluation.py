import json
import sys
import time
from dataclasses import dataclass

from .server import Server
from .utils import (
    END,
    apply_split,
    find_executable,
    formatter,
    parse_model_args,
    prepare_task_ds,
    run_command_with_progress,
    str_hash,
)


@dataclass
class Context:
    fa_flag: str
    model_arg_dict: dict[str,str]
    llama_args: str
    quiet: bool
    recalc: bool
    disable_ansi: bool
    server_cmd: str
    perplexity_cmd: str

    def get_model_args(self, model: str):
        for model_ss, ma in self.model_arg_dict.items():
            if model_ss in model:
                return ma
        return ''

    def server(self, model: str, args: str|list[str]|None=None) -> Server:
        args = args.split() if isinstance(args, str) else [] if not args else args
        return Server(
            self.server_cmd,
            model,
            (
                self.fa_flag.split() +
                ["-kvu", "-ngl", "99"] +
                args +
                self.get_model_args(model).split()
            )
        )

    @classmethod
    def from_args(cls, args):
        return Context(
            fa_flag="" if args.disable_flash_attention else "-fa on",
            model_arg_dict=parse_model_args(args.model_args),
            llama_args=args.llama_args,
            quiet=args.quiet,
            recalc=args.recalc,
            disable_ansi=args.disable_ansi,
            server_cmd=find_executable(args.llama_path, "llama-server"),
            perplexity_cmd=find_executable(args.llama_path, "llama-perplexity"),
        )

def run_ppl_task(ctx: Context, model: str, task: dict, model_archives, save_callback):
    model_args = ctx.get_model_args(model)
    dataset_invoc = prepare_task_ds(task)
    name = task['name']
    command = f"{ctx.fa_flag} -m {model} -kvu -ngl 99 {dataset_invoc} {task['llama_args']} {model_args}"
    if not ctx.quiet:
            print(END + f"\n{name}:")
    if not ctx.recalc and name in model_archives and isinstance(model_archives[name], list):
        return model_archives[name]

    left = task.get('left')
    start_time = time.time()
    split = None
    attempt = ''
    result = None
    try:
        attempt = f"{ctx.perplexity_cmd} {command} {ctx.llama_args}"
        result = run_command_with_progress(attempt, ansi=not ctx.disable_ansi)
    except Exception:
        pass
    if result:
        try:
            split = apply_split(result, left).split("\n", 1)[0]
        except ValueError:
            pass
    if split is None:
        print(f"Warning: unable to generate a split from any of the perplexity commands for task {name} and model {model}. Attempt: {attempt}")
        sys.exit(1)
    end_time = time.time()
    rv = [split, int(end_time - start_time)]
    model_archives[name] = rv
    save_callback()
    return rv

def run_thinking_task(ctx: Context, model: str, task: dict, model_archives: dict, save_callback):
    dataset_path = prepare_task_ds(task, path_only=True)
    name = task['name']
    if not ctx.quiet:
            print(END + f"\n{name}:")
    if not ctx.recalc and name in model_archives and isinstance(model_archives[name], list):
        return model_archives[name]

    if not dataset_path.endswith("jsonl"):
        raise NotImplementedError(f"unsupported file format: {dataset_path}")

    with open(dataset_path) as f:
        content = f.read()
    ds = []
    cache = model_archives.setdefault("_cache", {})
    found = 0
    for line in [x.strip() for x in content.split("\n") if x.strip()]:
        entry = json.loads(line)
        ds.append(entry)
        key = str_hash(f"{entry}")
        found += key in cache

    if not ctx.quiet:
        print(f"{len(ds)} entries in dataset ({found} found in cache, {len(ds)-found} generations left)")

    start_time = time.time()
    correct = 0
    total = 0
    with ctx.server(model, task['llama_args']) as server:
        for entry in ds:
            key = str_hash(f"{entry}")
            if key in cache:
                response = cache[key]
            else:
                task_verbalized = formatter(task['template'], entry)
                messages = [
                    {"role":"system","content":"You are an AI assistant. Think carefully before responding."},
                    {"role":"user", "content": task_verbalized.rstrip() + "\n\nRespond exactly as follows, without including reasoning or motivation:\n\nAnswer: LABEL."},
                ]
                print(f"Calling with {json.dumps(messages, indent=4)}")
                response = server.complete(messages, echo=True)
                cache[key] = response
                save_callback()

            response = response['response']
            # Canonicalize (Answer is **B**.)
            response = response.replace("*", "")
            for variant in {"The correct answer is", "The anwer is", "Correct Choice:\n"}:
                if variant + ": " in response:
                    response = "Answer" + response.split(variant, 1)[1]
                    break
                elif variant + " " in response:
                    response = "Answer: " + response.split(variant + " ", 1)[1]
                    break
                elif variant in response:
                    response = "Answer: " + response.split(variant, 1)[1]

            correct_label = "ABCDEFGHIJKLMNOP"[entry['answer']]
            if "Answer: " in response:
                answer_label = response.split("Answer: ", 1)[-1][0]
            else:
                breakpoint()
                raise ValueError("can't extract answer")
            if correct_label == answer_label:
                correct += 1
            total += 1
            print(f"[ {correct} / {total} : {correct/total:.2%} ]")
    end_time = time.time()
    rv = [correct / total, int(end_time - start_time)]
    model_archives[name] = rv
    save_callback()
    return rv
