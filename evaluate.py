#!/usr/bin/env python3
import argparse
import sys

import yaml

from tools.evaluation import Context, run_ppl_task, run_thinking_task
from tools.utils import (
    GRAY,
    MV_UP,
    WHITE,
    Archive,
    calc_eta,
    disable_ansi,
    get_model_size,
    parse_score,
    parse_task_list,
    process_score,
    sizestr,
    timestr,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on benchmarks.")
    parser.add_argument("models", nargs="+", help="Paths to GGUF model files to evaluate.")
    parser.add_argument("--llama_path", default="", help="Path to the llama.cpp project. If not given, llama-perplexity is assumed to be provided in shell path.")
    parser.add_argument("--quiet", action="store_true", help="Only output results, don't show progress.")
    parser.add_argument("--disable_ansi", action="store_true", help="Disable ANSI color codes in output.")
    parser.add_argument("--disable_sorting", action="store_true", help="Do not sort the outputs by the (aggregate) score before displaying them. Instead display them in the order they were given.")
    parser.add_argument("--disable_flash_attention", action="store_true", help="Do not use flash attention.")
    parser.add_argument("--llama_args", default="", help="Extra arguments to pass to the perplexity binary.")
    parser.add_argument("--recalc", action="store_true", help="Do not use cached values, if present.")
    parser.add_argument("--model_args", nargs="+", help="Provide model specific arguments, in the form <model search string>:<args>, e.g. --model_args GLM-4.5-Air:\"--n-cpu-moe 22\"")

    with open("tasks.yml") as f:
        tasks = yaml.load(f, Loader=yaml.FullLoader)
    task_list = ",".join(task['name'].lower() for task in tasks)
    parser.add_argument("--tasks", default="base", help=f"Tasks to run. Can be a list of {task_list}, base, extended, or a negative list such as except:{task_list}, which runs all tasks except those listed. base will run all tasks not marked as extended in tasks.yml. extended will run all tasks, including tasks marked extended, except tasks marked as extends by an extended task (e.g. MMLU-Redux-2.0 will not run by default if MMLU-Redux-2.0-Big runs).")

    args = parser.parse_args()
    ctx = Context.from_args(args)

    if args.disable_ansi:
        disable_ansi()

    tasks = parse_task_list(args.tasks, tasks)
    if len(tasks) == 0:
        print("No tasks. Great, we're done.")
        sys.exit(0)

    archives = Archive(".profile_archives.pk")

    # Sort models by file size
    model_list = args.models
    sizes = [get_model_size(m) for m in model_list]
    model_size = dict(zip(model_list, sizes))
    model_list = [m for _, m in sorted(zip(sizes, model_list), reverse=True)]

    final_results = []
    times = [0] * len(model_list)
    task_names = [task['name'] for task in tasks]
    task_fmt = "{task:<" + str(max(1 + len(tn) for tn in task_names)) + "}"
    for i, model in enumerate(model_list):
        if not args.quiet:
            print()
            etas = calc_eta(model_list, model_size, task_names, archives.content)
            for m, t in zip(model_list, times):
                col = WHITE if m == model else GRAY
                is_estimate = False
                if t < 1:
                    t = int(etas[m][0])
                    is_estimate = not etas[m][1]
                print(f"{col}{timestr(t, is_estimate, col)}{col} {sizestr(model_size[m]):>9} {m}")
        model_archives = archives.setdefault(model, {})

        model_time = 0
        def run(task: dict):
            executor = run_thinking_task if task.get('thinking', False) else run_ppl_task
            rv = executor(ctx, model, task, model_archives, archives.save)
            if not args.quiet:
                print(MV_UP(1) + task_fmt.format(task=task['name'] + ':') + f" {rv[0]}")
            nonlocal model_time
            model_time += rv[1]
            return rv[0]

        results = {}
        for task in tasks:
            if "scores" not in task:
                task["scores"] = []
            results[task['name']] = run(task)
            task["scores"].append(process_score(parse_score(results[task['name']]), task))

        times[i] = model_time

        final_results.append(results)

    # Normalize scores per task and calculate aggregates, then sort report output
    for i, (model, results) in enumerate(zip(model_list, final_results)):
        aggregated_score = 0
        for task in tasks:
            if 'min' not in task:
                task['max'] = max(task['scores'])
                task['winner'] = model_list[next(i for i, score in enumerate(task['scores']) if score == task['max'])]
            score = task['scores'][i]
            if score > 0:
                normalized_score = score / task['max']
                aggregated_score += normalized_score
        results['score'] = aggregated_score / len(tasks)

    if args.disable_sorting:
        # Simply calculate each task winner and move on
        for task in tasks:
            max_score = max(task['scores'])
            task['winner'] = model_list[next(i for i, score in enumerate(task['scores']) if score == max_score)]
    else:
        final_results = sorted(final_results, key=lambda r: r['score'], reverse=True)

    report = []
    for model, results in zip(model_list, final_results):
        score = results.pop('score')
        if len(tasks) > 1:
            next_report = "\n".join(task_fmt.format(task=name + ':') + f" {score}" for name, score in results.items())
            report.append(f"{model} : {score:.5f}\n{next_report}")
        else:
            report.append(f"{results[tasks[0]['name']]} : {model}")

    if len(tasks) > 1:
        sep = '\n' + '-' * max(len(f) for f in ('\n'.join(report)).split("\n")) + '\n'
    else:
        sep = '\n'
    print("==== FINAL REPORT ====\n\n" + sep.join(report).strip())

if __name__ == "__main__":
    main()
