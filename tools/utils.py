import os
import pickle
import subprocess
import sys
import warnings

import numpy as np


def ansii_mv_up(steps: int):
    return f"\033[{steps}A"

def nop_i(i: int):
    return ""

END   = "\033[0m"
BROWN = "\033[0;33m"
WHITE = "\033[1;97m"
GRAY  = "\033[0;37m"
CLR   = "\033[K"
MV_UP = ansii_mv_up

def disable_ansi():
    global END, BROWN, WHITE, GRAY, CLR, MV_UP
    END   = ""
    BROWN = ""
    WHITE = ""
    GRAY  = ""
    CLR   = ""
    MV_UP = nop_i

def handle_multipart(path: str):
    """
    Detect multipart files and yield each part.
    Properly handles 0-index content (huggingface models) and 1-index content (gguf files), but
    requires that the caller passes in the *first* file in the list.
    Does not do existence checks. I.e. if the filename is 00001-of-00050, it will yield all
    00001 through 00050 without checking file existence.
    Yields the path as is if it does not appear to be multipart.
    """
    if "0-of-" not in path and "1-of-" not in path:
        yield path
        return

    # Some are 1-based, some are 0-based. 1-based ones end at the of counter, 0-based end
    # one less.
    onebased = "00001-of-" in path
    start = "00001" if onebased else "00000"

    # Multi-file model
    # Format is: xxx-NNNNN-of-MMMMMyyy
    # We need MMMMM and then we can just iterate
    prefix, last = path.rsplit(start + "-of-", 1)
    # last = MMMMMyyy
    last, suffix = last[:5], last[5:]
    try:
        lastnum = int(last)
    except ValueError:
        print(f"Warning: file looks like a multi-part file but can't figure out the parts: {path}")
        yield path
        return

    for i in range(1 if onebased else 0, lastnum + (1 if onebased else 0)):
        mpath = f"{prefix}{i:05d}-of-{last}{suffix}"
        # Note: we do not check if mpath exists or not. We assume that if we have 00001-of-00003, then 1, 2, and 3 exist.
        yield mpath

def get_model_size(path: str) -> int:
    return sum(os.path.getsize(p) for p in handle_multipart(path))

def parse_task_list(tasks: str, obs: list[dict]) -> list[dict]:
    """
    Takes a list of tasks in the form "task1,task2,..." with an optional "except:" prefix, as
    well as a list of task objects, and generates a list of objects that match the task list.
    If the "except:" prefix is present, the filter is inverted (given tasks are excluded).
    """
    is_negative = tasks.startswith("except:")
    tasks = tasks.lower()
    if is_negative:
        tasks = tasks[7:]
    if tasks == "all":
        return [] if is_negative else obs
    taskset = set(tasks.split(","))
    return [task for task in obs if is_negative != (task['name'].lower() in taskset)]

def parse_model_args(margs: list[str]) -> dict[str, str]:
    rv = {}
    if not margs:
        return rv
    for ma in margs:
        model_substr, model_args = ma.split(":", 1)
        if model_substr in rv:
            rv[model_substr] += " " + model_args
        else:
            rv[model_substr] = model_args
    return rv

def apply_split(content: str, left: None|str|list[str]=None, right: None|str|list[str]=None) -> str:
    """
    Apply splits on content to form a result.
    The left split removes stuff on the "left" side of content. The right split removes stuff on the "right" side.
    If a list of strings, several splits are done in succession, one at a time.
    Example: left="/", right=".", content="/models/my-model-q8_0.gguf", result="my-model-q8_0"
    """
    if left is None and right is None:
        return content
    # Begin by left stripping
    if left is not None:
        for comp in left if isinstance(left, list) else [left]:
            if comp not in content:
                raise ValueError(f"Left split component '{comp}' not found in content.")
            content = content.rsplit(comp, 1)[-1]
    # Then right stripping
    if right is not None:
        for comp in right if isinstance(right, list) else [right]:
            if comp not in content:
                raise ValueError(f"Right split component '{comp}' not found in content.")
            content = content.split(comp, 1)[0]
    return content.strip()

def find_executable(path: str, file: str) -> str:
    """
    llama.cpp specific executable location finder, which checks common paths for file.
    """
    bin = path or file
    if not bin.endswith(file):
        # Look for variants; path/file, path/build/bin/file, path/bin/file
        for variant in ["/", "/build/bin/", "/bin/"]:
            bin = path + variant + file
            if os.path.exists(bin):
                break
        if not os.path.exists(bin):
            raise FileNotFoundError(f"Executable {file} not found in path {path}.")

    try:
        subprocess.run([bin, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Error: {file} binary not found. Provided path: {path}")
        sys.exit(1)
    return bin

def calc_eta(model_list: list[str],
             model_sizes: dict[str, int],
             tasks: list[str],
             archives: dict[str, dict[str, list]],
             poly_degree: int = 2) -> dict[str, tuple[float, bool]]:
    """
    Calculate ETA for remaining models/tasks using polynomial regression on model sizes.

    Args:
        model_list: List of all model names
        model_sizes: Dictionary mapping model names to their sizes in bytes
        tasks: List of task names
        archives: Nested dict with structure archives[model][task][1] containing execution times
        target_model: If specified, return ETA only for this model. If None, return for all models.
        poly_degree: Degree of polynomial to fit (default: 2 for quadratic)

    Returns:
        Dictionary with model to a tuple of ETA and an is-exact bool.
    """
    results = {}

    for model_name in model_list:
        model_result = {}
        model_size = model_sizes[model_name]

        completed_tasks = {}
        if model_name in archives:
            for task in tasks:
                if task in archives[model_name] and len(archives[model_name][task]) > 1:
                    completed_tasks[task] = archives[model_name][task][1]

        for task in tasks:
            if task in completed_tasks:
                model_result[task] = completed_tasks[task]
            else:
                eta = predict_task_time(task, model_size, model_list, model_sizes,
                                      archives, poly_degree)
                model_result[task] = eta

        if len(model_result) == len(tasks):
            all_completed = all(task in completed_tasks for task in tasks)
            results[model_name] = (sum(model_result.values()), all_completed)

    return results


def predict_task_time(task: str,
                     target_size: int,
                     model_list: list[str],
                     model_sizes: dict[str, int],
                     archives: dict[str, dict[str, list]],
                     poly_degree: int = 2) -> float:
    """
    Predict execution time for a specific task based on model size using polynomial regression.

    Args:
        task: Task name to predict
        target_size: Size of the model to predict for
        model_list: List of all model names
        model_sizes: Dictionary of model sizes
        archives: Archives containing timing data
        poly_degree: Degree of polynomial to fit

    Returns:
        Predicted execution time
    """
    sizes = []
    times = []

    for model in model_list:
        if (model in archives and
            task in archives[model] and
            len(archives[model][task]) > 1 and
            model in model_sizes):

            sizes.append(model_sizes[model])
            times.append(archives[model][task][1])

    if len(sizes) < 2:
        return estimate_fallback_time(target_size, model_list, model_sizes, archives)

    sizes = np.array(sizes)
    times = np.array(times)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs = np.polyfit(sizes, times, min(poly_degree, len(sizes) - 1))
        prediction = np.polyval(coeffs, target_size)
        return max(0.0, float(prediction))
    except (np.linalg.LinAlgError, ValueError):
        return estimate_fallback_time(target_size, model_list, model_sizes, archives)


def estimate_fallback_time(target_size: int,
                          model_list: list[str],
                          model_sizes: dict[str, int],
                          archives: dict[str, dict[str, list]]) -> float:
    """
    Fallback estimation when polynomial fitting fails.
    Uses linear scaling based on average time per byte.
    """
    total_time = 0
    total_size = 0

    for model in model_list:
        if model in archives and model in model_sizes:
            for task_times in archives[model].values():
                if len(task_times) > 1:
                    total_time += task_times[1]
                    total_size += model_sizes[model]

    if total_size > 0:
        avg_time_per_byte = total_time / total_size
        return avg_time_per_byte * target_size
    return 1.0


def run_command_with_progress(command, shell=True, ansi=True):
    if not ansi:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}\nOutput: {result.stdout}\nError: {result.stderr}")
        return result.stdout

    process = subprocess.Popen(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    output_lines = []
    current_line = ""
    displayed_lines = []
    first_display = True
    print(f"\n\n\n\n{GRAY}", end="")

    def update_display():
        nonlocal first_display
        if not first_display:
            lines_to_move_up = min(len(displayed_lines), 5)
            if lines_to_move_up > 0:
                print(MV_UP(lines_to_move_up), end='')
        for i, line in enumerate(displayed_lines[-5:]):
            print(f"{CLR}{line[:100]}")

        first_display = False

    assert process.stdout
    while True:
        try:
            char = process.stdout.read(1)
        except Exception:
            continue
        if not char:
            break

        if char == '\n':
            output_lines.append(current_line)
            displayed_lines.append(current_line)
            if len(displayed_lines) > 5:
                displayed_lines = displayed_lines[-5:]  # Keep only last 5
            update_display()
            current_line = ""
        else:
            current_line += char

    if current_line:
        output_lines.append(current_line)
        displayed_lines.append(current_line)
        if len(displayed_lines) > 5:
            displayed_lines = displayed_lines[-5:]
        update_display()

    return_code = process.wait()
    complete_output = '\n'.join(output_lines)

    if return_code == 0:
        if not first_display and displayed_lines:
            lines_shown = min(len(displayed_lines), 5)
            print(MV_UP(lines_shown), end='')
            for _ in range(lines_shown):
                print(CLR)
            print(MV_UP(lines_shown), end='')

        print(END, end="")
        return complete_output
    else:
        print("\nError:")
        print(complete_output)
        raise subprocess.CalledProcessError(return_code, command, complete_output)

def prepare_task_ds(task: dict, quiet: bool=False):
    if 'dataset_url' not in task:
        dataset = task['dataset']
        if not os.path.exists(f"datasets/{dataset}"):
            print(f"Dataset {dataset} not available, and no dataset URL was provided.")
            raise RuntimeError
    else:
        url = task['dataset_url']
        dataset = url.rsplit("/", 1)[-1]
        if not os.path.exists(f"datasets/{dataset}"):
            if not quiet:
                print(f"Downloading {dataset} from {url}...")
            os.makedirs("datasets", exist_ok=True)
            subprocess.run(f"wget -q -O datasets/{dataset} {url}", shell=True, check=True)
    flag = "-bf" if dataset.endswith(".bin") else "-f"
    return f"{flag} datasets/{dataset}"



def timestr(t: int, is_eta: bool=False) -> str:
    if t < 1:
        return "[--:--:--]"
    (L, R) = (BROWN, END) if is_eta else ("", "")

    return f"[{L}{t//3600:02}:{(t%3600)//60:02}:{(t%60):02}{R}]"

def sizestr(sz: int) -> str:
    if sz < 1025:
        return f"{sz} B"
    if sz < 1024 * 1025:
        return f"{sz/1024:.1f} kB"
    if sz < 1024 * 1024 * 1025:
        return f"{sz/(1024 * 1024):.1f} MB"
    return f"{sz/(1024 * 1024 * 1024):.1f} GB"

class Archive:
    def __init__(self, path: str):
        self.path = path
        try:
            with open(path, "rb") as f:
                self.content = pickle.load(f)
        except Exception:
            self.content = {}

    def save(self):
        with open(self.path + "_", "wb") as f:
            pickle.dump(self.content, f)
        os.rename(self.path + "_", self.path)

    def __contains__(self, key: str) -> bool:
        return key in self.content

    def __getitem__(self, key: str):
        return self.content[key]

    def __setitem__(self, key: str, value):
        self.content[key] = value

def parse_score(s: str|None) -> float:
    if s is None:
        return -1
    # TODO: Deal with y in "x +/- y"
    if "tasks):" in s:
        s = s.rsplit("tasks):", 1)[-1].strip()
    s = s.strip().split()[0]
    if s.endswith("%"):
        # We sometimes get scores output as 12.3400000 (ik_llama) and sometimes as 12.3400000% (llama)
        # -- if we are fancy-pants and convert to 0.1234 for the latter case, we end up with 100x score
        # for the former output, given the same score. So we just drop the %. The default aggregation
        # normalizes anyway so it doesn't make a diff either way, as long as we do the same thing
        # consistently.
        return float(s[:-1]) # / 100
    return float(s)


def traverse(d: dict|str|int|float, path: str):
    """
    Traverse and emit endpoints.
    Format:
        field1.field2.field3...
    """
    if not isinstance(d, dict):
        if path != '':
            breakpoint()
        assert path == ''
        yield d
        return
    comps = path.split(".", 1)
    curr, rem = comps[0], (comps[1] if len(comps) > 1 else '')
    if " or " in curr:
        alts = curr.split(" or ")
        for alt in alts:
            if alt.strip() in d:
                curr = alt.strip()
                break
    if curr not in d:
        raise KeyError
    value = d[curr]
    if isinstance(value, list) or isinstance(value, np.ndarray):
        for item in value:
            yield from traverse(item, rem)
        return
    if rem == '':
        yield value
        return
    yield from traverse(value, rem)

def ext_format(entry: dict, fmt: str, env: dict|None=None):
    """
    Traverse {entries} in fmt and yield results via traverse.
    """
    env = env or {"_idx":0}
    if not ('{' in fmt and '}' in fmt):
        yield fmt.replace("_LB", "{").replace("_RB", "}").format(**env)
        return
    lbracketed = fmt.split("{", 1)
    rbracketed = lbracketed[1].split("}", 1)
    key = rbracketed[0].strip()
    rekey = "id_" + str(env['_idx'])
    env['_idx'] += 1
    rewritten = lbracketed[0] + "_LB" + rekey + "_RB" + rbracketed[1]
    for value in traverse(entry, key):
        env[rekey] = value
        yield from ext_format(entry, rewritten, env)
    return
