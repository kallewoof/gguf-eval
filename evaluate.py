#!/usr/bin/env python3
import argparse
import sys
import time

import yaml

from tools.utils import (
    END,
    GRAY,
    MV_UP,
    WHITE,
    Archive,
    apply_split,
    calc_eta,
    disable_ansi,
    find_executable,
    get_model_size,
    parse_model_args,
    parse_score,
    parse_task_list,
    prepare_task_ds,
    run_command_with_progress,
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
    parser.add_argument("--tasks", default="all", help=f"Tasks to run. Can be a list of {task_list}, all, or a negative list such as except:{task_list}, which runs all tasks except those listed.")

    args = parser.parse_args()
    fa_flag = "" if args.disable_flash_attention else "-fa"
    model_arg_dict = parse_model_args(args.model_args)

    perplexity_cmd = find_executable(args.llama_path, "llama-perplexity")

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
        model_args = ''
        for model_ss, ma in model_arg_dict.items():
            if model_ss in model:
                model_args = ma
                break
        if not args.quiet:
            print()
            etas = calc_eta(model_list, model_size, task_names, archives.content)
            for m, t in zip(model_list, times):
                col = WHITE if m == model else GRAY
                is_estimate = False
                if t < 1:
                    t = int(etas[m][0])
                    is_estimate = not etas[m][1]
                print(f"{col}{timestr(t, is_estimate)}{col} {sizestr(model_size[m]):>9} {m}")
        if model in archives:
            model_archives = archives[model]
        else:
            model_archives = {}
            archives[model] = model_archives

        model_time = 0
        def run(task: dict):
            dataset_invoc = prepare_task_ds(task)
            name = task['name']
            command = f"{fa_flag} -m {model} -kvu -ngl 99 {dataset_invoc} {task['llama_args']} {model_args}"
            if not args.quiet:
                    print(END + f"\n{name}:")
            if not args.recalc and name in model_archives and isinstance(model_archives[name], list):
                rv = model_archives[name]
            else:
                left = task.get('left')
                start_time = time.time()
                split = None
                attempt = ''
                result = None
                try:
                    attempt = f"{perplexity_cmd} {command} {args.llama_args}"
                    result = run_command_with_progress(attempt, ansi=not args.disable_ansi)
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
                archives.save()
            if not args.quiet:
                print(MV_UP(1) + task_fmt.format(task=name + ':') + f" {rv[0]}")
            nonlocal model_time
            model_time += rv[1]
            return rv[0]

        results = {}
        for task in tasks:
            if "scores" not in task:
                task["scores"] = []
            results[task['name']] = run(task)
            task["scores"].append(parse_score(results[task['name']]))

        times[i] = model_time

        final_results.append(results)

    if args.disable_sorting:
        # Simply calculate each task winner and move on
        for task in tasks:
            max_score = max(task['scores'])
            task['winner'] = model_list[next(i for i, score in enumerate(task['scores']) if score == max_score)]
    else:
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
