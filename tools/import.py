# Import an existing dataset, converting it to json format compatible with gguf-eval

"""
Example usage:

For https://huggingface.co/datasets/deepmind/narrativeqa we have
{
    "document": {"id": "0123", "summary": {"text": "text summary"}, "title": "title", "text": "long text"},
    "question": {"text": "Who does Arabella Mason wed?"},
    "answers": [{"text": Ben Keene, Delmar's valet"}, "text":"Ben Keene"}]
}

Here, we choose to use only the summary, but we could use the text as well, if we had the context for it.
python tools/import.py \
    datasets/deepmind_narrativeqa/data/validation-00000-of-00003.parquet \
    deepmind_narrativeqa_summary.json \
    "{document.title or start} summary: {document.summary.text}\n\n{question.text}" \
    "{answers.text}" \
    --multiple_choice

We can easily use the entire text instead:
python tools/import.py \
    datasets/deepmind_narrativeqa/data/validation-00000-of-00003.parquet \
    deepmind_narrativeqa_summary.json \
    "{document.title or start}:\n{document.text}\n\n{question.text}" \
    "{answers.text}" \
    --multiple_choice
"""

import argparse
import json

from utils import ext_format, handle_multipart


def main():
    parser = argparse.ArgumentParser(description="Import an existing dataset and convert it to json format compatible with gguf-eval")
    parser.add_argument("input_file", type=str, help="Path to the input dataset file")
    parser.add_argument("output_file", type=str, help="Path to save the converted json file")
    parser.add_argument("question_format", type=str, default="{question}", help="Formatting to use for the question part. Formatting can be a path with dots. It may use the or keyword for alternatives, e.g. 'document.title or start' to use the document.title key if available, otherwise the document.start key.")
    parser.add_argument("answer_format", type=str, default="{answer}", help="Formatting to use for the answer part. Formatting can be a path with dots and supports arrays for multi-answer sources.")
    parser.add_argument("--multiple_choice", action="store_true", help="Whether the dataset is a multi-choice dataset (answers is an array of options). Currently only supports multiple correct answers, i.e. all answers are correct. If not set, the dataset is assumed to be a single-answer dataset with one correct answer and multiple wrong answers. Single answer datasets are not yet supported.")

    args = parser.parse_args()

    # De-escape formats which may have e.g. "\n" or similar in them
    question_format = args.question_format.replace("\\n", "\n").replace("\\t", "\t")
    answer_format = args.answer_format.replace("\\n", "\n").replace("\\t", "\t")

    final_data = []
    for file in handle_multipart(args.input_file):
        if file.endswith("parquet"):
            import pandas as pd
            # Load the input dataset using pandas
            df = pd.read_parquet(file)
            data = df.to_dict(orient='records')
        elif file.endswith("json"):
            # Load the input dataset as json
            with open(args.input_file) as infile:
                data = json.load(infile)
        else:
            raise ValueError(f"unknown input file format (expected .json or .parquet): {file}")
        final_data.extend(data)

    converted = []
    """
    {
        "multiple_correct": {
            "answers": ["Right answer", "Wrong answer1", ...],
            "labels": [1, 0, ...]
        },
        "question": "Question: \"What is the smallest country in the world that is at least one square mile in area?\" Answer:",
        "single_correct": {
            "answers": ["Single right answer", "Wrong answer1", ...]
            "labels": [1, 0, ...]
        }
    },
    """
    if args.multiple_choice:
        def convert_answer(entry: dict, out: dict):
            labels = []
            answers = []
            for answer in ext_format(entry, answer_format):
                answers.append(answer.strip())
                labels.append(1) # TODO: this assumes all answers are correct
            out["multiple_correct"] = {
                "answers": answers,
                "labels": labels
            }
    else:
        raise NotImplementedError
        # def convert_answer(entry: dict, out: dict):
        #     labels = []
        #     answers = []


    for entry in final_data:
        assert isinstance(entry, dict)
        question = next(ext_format(entry, question_format), None)
        if question is None:
            continue
        out = {
            "question": question.strip(),
            "multiple_correct": {},
            "single_correct": {}
        }
        convert_answer(entry, out)
        converted.append(out)

    # Save the converted dataset to the output file
    with open(args.output_file, 'w') as outfile:
        json.dump(converted, outfile, indent=2, ensure_ascii=False)

    print(f"Dataset imported and saved to {args.output_file}")

if __name__ == "__main__":
    main()
