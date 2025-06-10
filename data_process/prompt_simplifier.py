import json
import argparse
from collections import defaultdict
import tqdm

from collections import defaultdict

def create_system_prompt(content):
    new_content = "You are an expert in the game of Diplomacy. Diplomacy is a strategic board game set in the early 20th century, where players assume the roles of the major powers of Europe, each with their own distinct territories and military units. Your primary objective is to gain control over a specified number of supply centers, which serve as the power bases for your military units.\n\nThe current game time and phase can be found in [Game Time and Phase], and the current board state can be found in [Board State]. In [Board State], the regions occupied by each country will be shown in turn, and each region has the format Area(attribute 1, attribute 2, ...). Note that the troops attribute for all regions defaults to (troops non-buildable, troops non-removable, troops undislodged), which will be omitted in the default state. Use this information wisely to inform your strategies and decisions. Aim to form alliances, negotiate with other players, and outmaneuver your opponents to achieve dominance.\n\nRemember to consider the following [Strategies] as you play:\n- Diplomacy and Negotiation: Engage with other players to form alliances and negotiate terms. Trust and communication are key components of the game.\n- Strategic Planning: Carefully plan your moves, considering both short-term gains and long-term objectives. Anticipate the moves of other players and adapt accordingly.\n- Resource Management: Control and defend supply centers to maintain and expand your military forces.\n- Flexibility: Be prepared to adjust your strategy based on the evolving game state and the actions of other players.\n\nNow, generate the action in the correct format, considering the [board state], [game time and phase] and [Strategies]. Your response should be in the following format:\nThe recommended strategic actions are as follows:\n[Recommend Action ``````]\n- ``````\n- ``````\n...\n- ``````\n \nFor example: \nThe recommended strategic actions are as follows:\n[Recommend Action ```1```]\n- ```FLEET in St. Petersburg - South Coast MOVE to Bothnia```\n- ```ARMY in Moscow MOVE to Ukraine```\n- ```ARMY in Warsaw MOVE to Galicia```\n- ```FLEET in Sevastopol MOVE to Rumania```\n"
    country = content.split()[12]
    return new_content, country

def create_user_prompt(board_state_content, country):
    regions = board_state_content.strip().split("\n\n")
    occupation_stats = defaultdict(lambda: defaultdict(lambda: {'names': [], 'count': 0, 'attributes': []}))
    # print(regions)
    first_region = True
    second_region = False
    for region in regions:
        region_info = region.split("\n")
        if second_region:
            region_name = region_info[2].lstrip("[").lstrip().strip("[]")
            second_region = False
        else:
            region_name = region_info[0].strip("[]")
        if first_region:
            game_time = region_info[1].lstrip()
            first_region = False
            second_region = True
        # print("region_info", region_info)
        # print("region_name", region_name)
        power = None  # 用于存储当前区域的权力归属
        attributes = []
        for line in region_info[0:]:
            # print(line)
            # input()
            if "[Board State]" in line:
                continue
            if "Not occupied by" in line:
                power = "UNOCCUPIED AREAS"
            elif "Occupied by" in line:
                # 确保正确提取占领者名称
                power = line.split("'s ")[0]
                power = ' '.join(line.split()[2:]).rstrip(".")

            # 根据区域描述添加属性
            if "Can build new troop from here" in line:
                attributes.append("troops buildable")
            elif "The troop here can be removed" in line:
                attributes.append("troops removable")
            elif "Dislodged" in line:
                attributes.append(line)

            if "This area is" in line and len(line.split()) == 4:
                line_split = line.split()
                attributes.append(line_split[-1].rstrip('.'))
            elif "This area is" in line and len(line.split()) > 5:
                if "neither" in line:
                    attributes.append("Not supply or home center")
                elif "home center" in line:
                    attributes.append(line[13:])
                elif "This area is a supply center" in line:
                    attributes.append("supply center") 

        if attributes:
            occupation_stats[power][region_name]['attributes'] = ', '.join(attributes)
            occupation_stats[power][region_name]['names'].append(region_name)
            occupation_stats[power][region_name]['count'] += 1

    user_prompt = f"[Your Country]:\n{country}\n\n[Game Time and Phase]:\n{game_time}\n\n[Board State]:"

    for power, regions in reversed(list(occupation_stats.items())):
        user_prompt += f"\n\n{power}:"
        for region_name, stats in regions.items():
            # 假设 attributes 是一个字符串，如果不是，需要相应地转换
            user_prompt += f"\n{region_name} ({stats['attributes']})"

    # print(user_prompt)
    return user_prompt.rstrip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="From json to json")
    parser.add_argument("--input_file_path", type=str, default='/home/xukaixuan/diplomacy_experiments/Prompt/prompt_html_medium_0716_100k.json')
    parser.add_argument("--output_file_path", type=str, default='/home/xukaixuan/diplomacy_experiments/Prompt/prompt_html_medium_0716_short_100k.json')

    args = parser.parse_args()

    # 假设 JSON 文件名为 'data.json'
    new_message = []
    
    # 打开 JSON 文件并读取数据
    with open(args.input_file_path, 'r', encoding='utf-8') as file:
        print(f"loading json file from {args.input_file_path}")
        messages = json.load(file)  # 加载 JSON 数据
        print("dataset lode done")

    for message_block in tqdm.tqdm(messages, desc="simplify prompt"):
        # 遍历每个消息块中的messages列表
        for message in message_block["messages"]:
            # 检查消息的角色
            # print(message)
            if message["role"] == "system":
                content = message.get("content")
                system_prompt, country = create_system_prompt(content)
            elif message["role"] == "user":
                content = message.get("content")
                user_prompt = create_user_prompt(content, country)
                # print(user_prompt)
            elif message["role"] == "assistant":
                order_prompt = message.get("content")
        
    with open(args.output_file_path, "+w") as f:
            json.dump(new_message, f, indent=2)
