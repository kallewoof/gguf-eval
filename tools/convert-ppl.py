import argparse
import json

from nltk.tokenize.treebank import TreebankWordDetokenizer


# from sacremoses import MosesDetokenizer

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Input file.")
parser.add_argument("output", type=str, help="Output file.")
args = parser.parse_args()

# detok = MosesDetokenizer()
detok = TreebankWordDetokenizer()

if args.input.endswith("jsonl"):
    values = []
    with open(args.input) as f:
        for line in f.read().strip().split("\n"):
            values.append(json.loads(line))

    with open(args.output, "w") as f:
        ob = values[0]
        if set(ob.keys()) == {"domain", "text"}:
            # LAMBADA
            for ob in values:
                text = ob['text'] #"`` i hope i did n't wake you , '' she said ."
                tokens = text.split()
                result = detok.detokenize(tokens)
                f.write(result + '\n')
