import argparse
import json

parser = argparse.ArgumentParser(description="truncate bin file")
parser.add_argument("--in_file", type=str, required=False, default="/home/xukaixuan/diplomacy_experiments/Prompt/high_0719/prompt_html_high_full.json")
parser.add_argument("--number", type=int, required=False, default=100000)
parser.add_argument("--out_file", type=str, required=False, default="/home/xukaixuan/diplomacy_experiments/Prompt/high_0719/prompt_html_high_100k.json")

args = parser.parse_args()

print(f"loading {args.in_file}")
try:
    with open(args.in_file, "r") as f:
        data = json.load(f)
except json.decoder.JSONDecodeError as e:
    with open(args.in_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

print(f"writing to {args.out_file}")
with open(args.out_file, "w") as f:
    if len(data) >= args.number:
        json.dump(data[:args.number], f, ensure_ascii=False, indent=4)
    else:
        json.dump(data, f, ensure_ascii=False, indent=4)