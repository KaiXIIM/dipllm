import os
from bs4 import BeautifulSoup
import cairosvg
import imgkit
import re
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--in_file", type=str, default="3004.html", help="输入的HTML文件路径")
arg_parser.add_argument("--out_folder", type=str, default="output_svgs", help="输出SVG文件夹路径")

args = arg_parser.parse_args()

html_file = args.in_file
output_dir = args.out_folder

os.makedirs(output_dir, exist_ok=True)

with open(html_file, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

svg_tags = soup.find_all('svg')

for i, svg_tag in enumerate(svg_tags):
    if i % 2 == 0:
        svg_str = str(svg_tag)
        for r in re.findall("^.*class=\"currentnoterect\".*$", svg_str, re.M):
            svg_str = svg_str.replace(r, "")
        for r in re.findall("^.*id=\"CurrentNote\".*$", svg_str, re.M):
            svg_str = svg_str.replace(r, "")
        imgkit.from_string(svg_str, f'{output_dir}/image_{int(i/2)}.png')

print(f'所有SVG标签已转换并保存到{output_dir}目录下。')