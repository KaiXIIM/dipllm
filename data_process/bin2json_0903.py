# INPUT_FILE_PATH = "/home/xukaixuan/diplomacy_experiments/results/diplomacy/adhoc/2024-07-31T233649.703237/c04_exploit/research_20240730_collect_expert_data_br/lau.loc.use@true_b2bca929/buffer1.bin"
INPUT_FILE_PATH = "/root/lsc/high_dataset_0721/c04_exploit/research_20240409_collect_expert_data/lau.loc.use@true_b2bca929/buffer1.bin"
OUTPUT_FILE_PATH = "/root/chai/sft/dataset/new_prompt_actor_0904.json"
MAX_DATA_COUNT = 1000
MAX_POSSIBLE_ORDER_NUM  = 5

from fairdiplomacy.selfplay import rela
import torch
import json
import tqdm
import re
import argparse


from collections import defaultdict


with open("data_process/new_order_vocabulary_nl_2.json", "r") as f:
    index2order = json.load(f)

playing_countries = {
    "Austria", "England", "France", "Germany",
    "Italy", "Russia", "Turkey"
}

index2country_str = {
    0: "Austria", 1: "England", 2: "France", 3: "Germany",
    4: "Italy", 5: "Russia", 6: "Turkey"
}

index2area_str_full = {
    0: "York", 1: "Edinburgh", 2: "London", 3: "Liverpool", 4: "North Sea",
    5: "Wales", 6: "Clyde", 7: "Norwegian Sea", 8: "English Channel", 9: "Irish Sea",
    10: "North Atlantic Ocean", 11: "Belgium", 12: "Denmark", 13: "Heligoland Bight", 14: "Holland",
    15: "Norway", 16: "Skagerrak", 17: "Barents Sea", 18: "Brest", 19: "Mid Atlantic Ocean",
    20: "Picardy", 21: "Burgundy", 22: "Ruhr", 23: "Baltic Sea", 24: "Kiel",
    25: "Sweden", 26: "Finland", 27: "St. Petersburg", 28: "St. Petersburg's North Coast", 29: "Gascony",
    30: "Paris", 31: "North Africa", 32: "Portugal", 33: "Spain", 34: "Spain's North Coast",
    35: "Spain's South Coast", 36: "Western Mediterranean", 37: "Marseilles", 38: "Munich", 39: "Berlin",
    40: "Bothnia", 41: "Livonia", 42: "Prussia", 43: "St. Petersburg's South Coast", 44: "Moscow",
    45: "Tunisia", 46: "Lyon", 47: "Tyrrhenian Sea", 48: "Piedmont", 49: "Bohemia",
    50: "Silesia", 51: "Tyrol", 52: "Warsaw", 53: "Sevastopol", 54: "Ukraine",
    55: "Ionian Sea", 56: "Tuscany", 57: "Naples", 58: "Rome", 59: "Venice",
    60: "Galicia", 61: "Vienna", 62: "Trieste", 63: "Armenia", 64: "Black Sea",
    65: "Rumania", 66: "Adriatic Sea", 67: "Aegean Sea", 68: "Albania", 69: "Apulia",
    70: "Eastern Mediterranean", 71: "Greece", 72: "Budapest", 73: "Serbia", 74: "Ankara",
    75: "Smyrna", 76: "Syria", 77: "Bulgaria", 78: "Bulgaria's East Coast", 79: "Constantinople",
    80: "Bulgaria's South Coast"
}
index2area_str = {
    0: "YOR", 1: "EDI", 2: "LON", 3: "LVP", 4: "NTH",
    5: "WAL", 6: "CLY", 7: "NWG", 8: "ENG", 9: "IRI",
    10: "NAO", 11: "BEL", 12: "DEN", 13: "HEL", 14: "HOL",
    15: "NWY", 16: "SKA", 17: "BAR", 18: "BRE", 19: "MAO",
    20: "PIC", 21: "BUR", 22: "RUH", 23: "BAL", 24: "KIE",
    25: "SWE", 26: "FIN", 27: "STP", 28: "STP_NC", 29: "GAS",
    30: "PAR", 31: "NAF", 32: "POR", 33: "SPA", 34: "SPA_NC",
    35: "SPA_SC", 36: "WES", 37: "MAR", 38: "MUN", 39: "BER",
    40: "BOT", 41: "LVN", 42: "PRU", 43: "STP_SC", 44: "MOS",
    45: "TUN", 46: "LYO", 47: "TYS", 48: "PIE", 49: "BOH",
    50: "SIL", 51: "TYR", 52: "WAR", 53: "SEV", 54: "UKR",
    55: "ION", 56: "TUS", 57: "NAP", 58: "ROM", 59: "VEN",
    60: "GAL", 61: "VIE", 62: "TRI", 63: "ARM", 64: "BLA",
    65: "RUM", 66: "ADR", 67: "AEG", 68: "ALB", 69: "APU",
    70: "EAS", 71: "GRE", 72: "BUD", 73: "SER", 74: "ANK",
    75: "SMY", 76: "SYR", 77: "BUL", 78: "BUL_EC", 79: "CON",
    80: "BUL_SC"
}
order_types = {
    "moves", "supports", "retreats", "holds", "convoys", "hold"
}

def get_unit_position_and_order(power_order_components):
    for order_type in order_types:
        if order_type in power_order_components:
            power_order_components_split = power_order_components.split(order_type)
            return power_order_components_split[0].strip(), f"{order_type}{power_order_components_split[1]}"
    return None, None
            

def find_unit_order(power_unit_postion, power_order_components):
    for unit_order in power_order_components:
        unit_position, unit_order_without_position = get_unit_position_and_order(unit_order)
        if unit_position is not None:
            if power_unit_postion in unit_position:
                return unit_order_without_position
    return "holds"


def get_specify_str(user_prompt, sorted_items, specify_str):
    for power, regions in sorted_items:
        if 'Areas Without Unit' in power:  # 确保不重复处理特定的 power
            # user_prompt += f"{power}:"
            user_prompt += f"\n{specify_str}:"
            sorted_regions = sorted(regions.items(), key=lambda item: item[0])
            is_unoccupied_supply_center_exist = False
            for region_name, stats in sorted_regions:
                if specify_str  in stats['attributes']:
                    is_unoccupied_supply_center_exist = True
                    stats['attributes'] = stats['attributes'].replace('unoccupied supply center', '')
                    if len(stats['attributes']) == 0:
                        user_prompt += f"\n{region_name}"
                    else:
                        user_prompt += f"\n{region_name} ({stats['attributes']})"
            if not is_unoccupied_supply_center_exist:
                user_prompt += f"\nNone"
            # user_prompt += "\n"
    return user_prompt

def create_user_prompt(board_state_content, country):
    regions = board_state_content.strip().split("\n\n")
    occupation_stats = defaultdict(lambda: defaultdict(lambda: {'names': [], 'count': 0, 'attributes': []}))
    first_region = True
    power_unit_postion = {}
    for power in playing_countries:
        power_unit_postion[power] = []
    for region in regions:
        if "'s North Coast" in region or "'s South Coast" in region or "s East Coast" in region  or "s Eastern Coast" in region or "'s Southern Coast" in region:
            continue
        region_info = region.split("\n")
    
        if first_region:
            game_time = region_info[1].lstrip().title()
            region_info = region_info[3:]
            first_region = False
            # region_name = region_info[3].lstrip("[").lstrip().strip("[]")
        region_name = region_info[0].strip("[]").lstrip().lstrip("[")
        attributes = []
        if "St. Petersburg" in region_name:
            attributes.append(f"including North and South Coast") 
        elif "Spain" in region_name:
            attributes.append(f"including North and South Coast") 
        elif "Bulgaria" in region_name:
            attributes.append(f"including East and South Coast") 
        power = None  # 用于存储当前区域的权力归属
        
        occupied_power = None
        power_center = None
        for line in region_info:
            if "[Board State]" in line:
                continue
            if "not occupied by anyone" in line:
                power = "Areas Without Unit"
            elif "Occupied by" in line:
                # 确保正确提取占领者名称
                power = line.split("'s ")[0]
                power = ' '.join(line.split()[2:]).rstrip(".")
                power_center = line.split("'s ")[0].split(' ')[-1].rstrip(".")
                power_unit_postion[power.split("'")[0]].append(f"{power.split(' ')[1]} in {region_name}")
            # 根据区域描述添加属性:
            # if "Can build new troop from here" in line:
            #     attributes.append("troops buildable")
            # elif "the troop here can be removed" in line:
            #     attributes.append("troops removable")
            if "not dislodged troop" not in line and "dislodged" in line:
                attributes.append(line)
            if "this area is" in line and len(line.split()) == 4 and "coast" not in line:
                line_split = line.split()
                attributes.append(line_split[-1].rstrip('.'))
            elif "this area is" in line and len(line.split()) > 5:
                # if "home center" in line and "neither" not in line:
                #     index = line.find("occupied by")
                #     occupied_power= line[index+len("occupied by"):].strip().strip(".").split(" ")[1]
                #     attributes.append(f"{occupied_power} supply center") 
                if "this area is a supply center, currently occupied by" in line:
                    index = line.find("occupied by")
                    occupied_power= line[index+len("occupied by"):].strip(".").strip() + "'s"
                    attributes.append(f"{occupied_power} supply center") 
                elif "this area is a supply center, currently not occupied" in line:
                    attributes.append("unoccupied supply center") 
        
            
        occupation_stats[power][region_name]['attributes'] = ', '.join(attributes)
        occupation_stats[power][region_name]['names'].append(region_name)
        occupation_stats[power][region_name]['count'] += 1
        if occupied_power is not None and occupied_power.split("'")[0] != power_center:
            occupation_stats[f"{occupied_power} center without units"][region_name]['names'].append(region_name)
            occupation_stats[f"{occupied_power} center without units"][region_name]['attributes'] = ', '.join(attributes)


    user_prompt = f"[Game Time and Phase]:\n{game_time}\n\n[Board State]:\n"
    user_prompt = f"[Your Power]:\n{country}\n\n[Game Time and Phase]:\n{game_time}\n\n[Board State]:\n"
    # 填充空白数据
    for power in playing_countries:
        power_army = f"{power}'s army"
        power_fleet = f"{power}'s fleet"
        power_center_without_units = f"{power}'s center without units"
        if power_army not in occupation_stats.keys():
            occupation_stats[power_army]["None"]['names'].append("None")
        if power_fleet not in occupation_stats.keys():
            occupation_stats[power_fleet]["None"]['names'].append("None")
        if power_center_without_units not in occupation_stats.keys():
            occupation_stats[power_center_without_units]["None"]['names'].append("None")

    country = country.title()
    sorted_items = sorted(occupation_stats.items(), key=lambda item: item[0])
    country_army = f"{country}'s army"
    country_fleet = f"{country}'s fleet"
    country_army_center = f"{country}'s center without units"
    specific_indices = [
        next((index for index, (p, _) in enumerate(sorted_items) if p == country_army)),
        next((index for index, (p, _) in enumerate(sorted_items) if p == country_fleet)),
        next((index for index, (p, _) in enumerate(sorted_items) if p == country_army_center))
    ]
    ignore_powers = []
    user_prompt += "Your Power Unit:\n"
    for specific_index in specific_indices:
        # if specific_index is None:
        #     user_prompt += f"{power}:\nNone\n"
        #     continue
        power, regions = sorted_items[specific_index]
        ignore_powers.append(power.split("'")[0])
        user_prompt += f"{power}:"
        sorted_regions = sorted(regions.items(), key=lambda item: item[0])
        for region_name, stats in sorted_regions:
            if len(stats['attributes']) == 0 or "'s center without units" in power:
                user_prompt += f"\n{region_name}"
            else:
                user_prompt += f"\n{region_name} ({stats['attributes']})"
        user_prompt += "\n"

    # 然后处理其他的 power
    user_prompt = user_prompt.rstrip('\n') + "\n\nOther Power Unit:"
    # user_prompt = user_prompt.rstrip() + ('\n')
    for index, (power, regions) in enumerate(sorted_items):
        if power.split("'")[0] not in ignore_powers and 'Areas Without Unit' not in power:  # 确保不重复处理特定的 power
            ignore_powers.append(power.split("'")[0])
            # user_prompt += f"{power}:"
            # sorted_regions = sorted(regions.items(), key=lambda item: item[0])
            user_prompt += f"\n{sorted_items[index][0]}:"
            sorted_regions = sorted(sorted_items[index][1].items(), key=lambda item: item[0])
            for region_name, stats in sorted_regions:
                if len(stats['attributes']) == 0:
                    user_prompt += f"\n{region_name}"
                else:
                    user_prompt += f"\n{region_name} ({stats['attributes']})"
            user_prompt += "\n"
            if index+2 < len(sorted_items):
                user_prompt += f"{sorted_items[index+2][0]}:"
                sorted_regions = sorted(sorted_items[index+2][1].items(), key=lambda item: item[0])
                for region_name, stats in sorted_regions:
                    if len(stats['attributes']) == 0:
                        user_prompt += f"\n{region_name}"
                    else:
                        user_prompt += f"\n{region_name} ({stats['attributes']})"
                user_prompt += "\n"

            if index+1 < len(sorted_items):
                user_prompt += f"{sorted_items[index+1][0]}:"
                sorted_regions = sorted(sorted_items[index+1][1].items(), key=lambda item: item[0])
                for region_name, stats in sorted_regions:
                    user_prompt += f"\n{region_name}"
                user_prompt += "\n"

    #  unoccupied supply center
    for power, regions in sorted_items:
        if 'Areas Without Unit' in power:  # 确保不重复处理特定的 power
            user_prompt += f"\n{power}:"
            user_prompt += "\nunoccupied supply center:"
            sorted_regions = sorted(regions.items(), key=lambda item: item[0])
            is_unoccupied_supply_center_exist = False
            for region_name, stats in sorted_regions:
                if "unoccupied" in stats['attributes']:
                    is_unoccupied_supply_center_exist = True
                    stats_attribute = stats['attributes'].replace('unoccupied supply center', '')
                    if len(stats_attribute) == 0:
                        user_prompt += f"\n{region_name}"
                    else:
                        user_prompt += f"\n{region_name} ({stats_attribute})"
            if not is_unoccupied_supply_center_exist:
                user_prompt += f"\nNone"
            # user_prompt += "\n"

    #  occupied supply center:
    user_prompt += "\n\n"
    for power, regions in sorted_items:
        if 'Areas Without Unit' in power:  # 确保不重复处理特定的 power
            is_occupied_supply_center_exist = False
            user_prompt += "occupied supply center:"
            sorted_regions = sorted(regions.items(), key=lambda item: item[0])
            for region_name, stats in sorted_regions:
                if "supply center" in stats['attributes'] and "unoccupied" not in stats['attributes']:
                    is_occupied_supply_center_exist = True
                    stats_attribute = stats['attributes'].replace(' supply center', '')
                    if len(stats_attribute) == 0:
                        user_prompt += f"\n{region_name}"
                    else:
                        user_prompt += f"\n{region_name} ({stats_attribute})"
            if not is_occupied_supply_center_exist:
                user_prompt += f"\nNone"
    
    for power, regions in sorted_items:
        if 'Areas Without Unit' in power:  # 确保不重复处理特定的 power
            is_not_supply_center_exist = False
            user_prompt += "\n\nnot supply center:"
            sorted_regions = sorted(regions.items(), key=lambda item: item[0])
            for region_name, stats in sorted_regions:
                if "supply center" not in stats['attributes']:
                    is_not_supply_center_exist = True
                    stats_attribute = stats['attributes'].replace("supply center", '')
                    if len(stats_attribute) == 0:
                        user_prompt += f"\n{region_name}"
                    else:
                        user_prompt += f"\n{region_name} ({stats_attribute})"
            if not is_not_supply_center_exist:
                user_prompt += f"\nNone"

    for key in power_unit_postion:
        power_unit_postion[key].sort()

    return user_prompt.rstrip().rstrip("\n"), power_unit_postion

def get_last_order_act(last_order_act, power_user_content, year_str, season_str, phase_type_str):
    if year_str == "1901" and season_str == "SPRING" and phase_type_str == "Diplomacy":
        return "None"
    last_order_act_str = ""
    if  len(last_order_act) == 0:
        return "None"
    your_power = power_user_content.split("\n")[1]
    last_order_act_regions = last_order_act.split("\n\n")
    for last_order_act_region in last_order_act_regions:
        if your_power in last_order_act_region:
            last_order_act_str += f"Your Power Order:\n{last_order_act_region}"
            break
    last_order_act_str += f"\n\nOther Power Order:\n"
    for last_order_act_region in last_order_act_regions:
        if your_power not in last_order_act_region:
            last_order_act_str += f"{last_order_act_region}\n"
        # last_order_act_str += "\n"
        
    return last_order_act_str.strip() + '\n'
            

def area_tensor2area_description_str(area_state):
    troop_list = [
        "army", "fleet"
    ]
    country_list = [
        "Austria", "England", "France", "Germany",
        "Italy", "Russia", "Turkey"
    ]
    area_type_list = [
        "land",
        "water",
        "coast"
    ]
    def get_troop(state):
        for idx, troop in enumerate(troop_list):
            if state[0 + idx] == 1:
                return troop
        return None
    def get_troop_country(state):
        for idx, country in enumerate(country_list):
            if state[2 + idx] == 1 and country in playing_countries:
                return country
        return None
    def can_build_troop(state):
        return state[9] == 1
    def troop_removable(state):
        return state[10] == 1
    def get_dislodged_troop(state):
        for idx, troop in enumerate(troop_list):
            if state[11 + idx] == 1:
                return troop
        return None
    def get_dislodged_troop_country(state):
        for idx, country in enumerate(country_list):
            if state[13 + idx] == 1 and country in playing_countries:
                return country
        return None
    def get_area_type(state):
        for idx, area_type in enumerate(area_type_list):
            if state[20 + idx] == 1:
                return area_type
        return None
    def get_supply_center_country(state):
        for idx, country in enumerate(country_list):
            if state[23 + idx] == 1 and country in playing_countries:
                return country
        return None
    def is_unoccupied_supply_center(state):
        return state[30] == 1
    def get_home_center_country(state):
        for idx, country in enumerate(country_list):
            if state[31 + idx] == 1 and country in playing_countries:
                return country
        return None

    description = ""
    if get_troop(area_state) is not None and get_troop_country(area_state) is not None:
        description += f"Occupied by {get_troop_country(area_state)}'s {get_troop(area_state)}.\n"
    else:
        description += "not occupied by anyone.\n"
    if can_build_troop(area_state):
        description += "Can build new troop from here.\n"
        
    else:
        description += "can not build new troop from here.\n"
    if troop_removable(area_state):
        description += "the troop here can be removed.\n"
    else:
        description += "the troop here can not be removed.\n"
    if get_dislodged_troop(area_state) is not None:
        description += f"{get_dislodged_troop_country(area_state)}'s {get_dislodged_troop(area_state)} was dislodged here\n"
    else:
        description += "not dislodged troop here\n"
    description += f"this area is {get_area_type(area_state)}.\n"
    # if get_home_center_country(area_state) is not None:
    #     description += f"this area is {get_home_center_country(area_state)}'s home center.\n"
    if get_supply_center_country(area_state) is not None:
        description += f"this area is a supply center, currently occupied by {get_supply_center_country(area_state)}.\n"
    elif is_unoccupied_supply_center(area_state):
        description += f"this area is a supply center, currently not occupied.\n"
    else:
        description += f"this area is neither a supply center nor a home center.\n"
    return description

def year2str(year):
    # year is an int tensor
    return str(int(year))

def season2str(season):
    if season[0] == 1:
        return "SPRING"
    elif season[1] == 1:
        return "FALL"
    elif season[2] == 1:
        return "SUMMER or WINTER"
    else:
        raise Exception(f"Invalid season: {season}")

def phase2str(phase_type):
    if phase_type == 65:
        return "Builds"
    elif phase_type == 77:
        return "Diplomacy"
    elif phase_type == 82:
        return "Retreats"
    else:
        raise Exception(f"Invalid phase type: {phase_type}")

def board_state2state_description_str(state):
    description = ""
    for area_idx in range(state.size(0)):
        area_name = index2area_str_full[area_idx]
        area_tensor = state[area_idx, :]
        area_description = area_tensor2area_description_str(area_tensor)
        description += f"[{area_name}]\n{area_description}\n"
    description = description.strip("\n")
    return description

def orders2str(orders, possible_action):
    order_idxs = set()  # 使用集合确保索引唯一
    for idx in range(orders.size(0)):
        order_idx = possible_action[idx][orders[idx]]
        if int(order_idx) != -1:
            order_idxs.add(order_idx)
    
    orders_strs = [f"{index2order[str(int(order_idx))]}\n" for order_idx in order_idxs]  # 生成订单字符串列表
    sorted_orders_strs = sorted(orders_strs)  # 对订单字符串进行排序
    sorted_orders_strs = "".join(sorted_orders_strs)  # 连接字符串
        
    # orders_str = "".join(orders_strs)  # 连接字符串
    if len(sorted_orders_strs) == 0:
        sorted_orders_strs = "None\n"
    sorted_orders_strs = sorted_orders_strs.replace(";", "\n")
    return sorted_orders_strs  # 移除尾部的换行符

def orders2list(orders, possible_action):
    order_idxs = set()  # 使用集合确保索引唯一
    for idx in range(orders.size(0)):
        order_idx = possible_action[idx][orders[idx]]
        if int(order_idx) != -1:
            order_idxs.add(order_idx)
    
    orders_strs = [f"{index2order[str(int(order_idx))]}" for order_idx in order_idxs]  # 生成订单字符串列表
    sorted_orders_strs = sorted(orders_strs)  # 对订单字符串进行排序

    if len(sorted_orders_strs) == 0:
        sorted_orders_strs = "None\n"

    return sorted_orders_strs  # 移除尾部的换行符

def get_action_component(input_string):
    # 将输入字符串按[ACTION]分割
    actions = re.split(r'\[ACTION\]\s*', input_string)
    actions = [action.strip() for action in actions if action.strip()]  # 移除空字符串

    all_actions = []
    for action in actions:
        # 分割每个动作为单独的行
        moves = action.split('\n')
        moves = [move.strip() for move in moves if move.strip()]  # 移除空行
        all_actions.append(moves)
        # 遍历子列表中的每个动作
    orders = []
    for action_group in all_actions:
    # 遍历子列表中的每个动作
        pre_orders = ''
        for action in action_group:
            orders.append(f"{pre_orders}{action}")
            pre_orders += f"{action}\n"
    from collections import OrderedDict

    unique_actions = list(OrderedDict.fromkeys(orders))
    return unique_actions

def country_orders2str(country_search_policy_order, country_search_policy_prob, country_possible_action, num_actions=1):
    combined_order_str = ""
    largest_prob_indices = torch.topk(country_search_policy_prob, k=num_actions, largest=True).indices.long()
    count = 1
    for idx in largest_prob_indices: 
        combined_order_str += f"[ACTION]\n"
        order = country_search_policy_order[idx, :]
        combined_order_str += orders2str(order, country_possible_action)
        combined_order_str += "\n"
        count += 1
    return combined_order_str.rstrip()

def country_possible_orders(country_search_policy_order, country_search_policy_prob, country_possible_action, num_actions=1):
    largest_prob_indices = torch.topk(country_search_policy_prob, k=num_actions, largest=True).indices.long()
    count = 1
    country_possible_orders_list = []
    orders_component_list = []
    set_orders = set() 
    for idx in largest_prob_indices: 
        order = country_search_policy_order[idx, :]
        country_possible_orders_list.append(orders2list(order, country_possible_action))
        count += 1

    country_possible_orders_list_truncate = []
    for index, orders in enumerate(country_possible_orders_list):
        if 'None' in orders:
            if index == 0:
                return ['None']
            break
    country_possible_orders_list_truncate = country_possible_orders_list[:index]  
    if country_possible_orders_list_truncate[0] == 'None':
        return ['None']
    
    lengths = [len(orders) for orders in country_possible_orders_list_truncate]
    shortest_length = min(lengths)

    for idx in range(shortest_length):
        for orders in country_possible_orders_list_truncate:
            set_orders.add(orders[idx])
        orders_component_list.append(list(set_orders))
        set_orders.clear()

    orders_component_list = sorted(orders_component_list)

    return orders_component_list

def parse_actions(input_string, num_actions):
    # 将输入字符串按[ACTION]分割
    actions = re.split(r'\[ACTION\]\s*', input_string)
    actions = [action.strip() for action in actions if action.strip()]  # 移除空字符串

    all_actions = []
    for action in actions:
        # 分割每个动作为单独的行
        moves = action.split('\n')
        moves = [move.strip() for move in moves if move.strip()]  # 移除空行
        all_actions.append(moves)
        # 遍历子列表中的每个动作
    orders = set() 
    for action_group in all_actions:
    # 遍历子列表中的每个动作
        pre_orders = ''
        for action in action_group:
            orders.add(f"{pre_orders}{action}")
            pre_orders += f"{action}\n"
    return orders

def get_power_sequence_messages(
        year_str, 
        season_str, 
        phase_type_str, 
        all_power_unit_postions, 
        all_power_order_components, 
        user_content, 
        last_order_act,
        all_power_possble_orders_list
    ):
    if phase_type_str == "Builds" or phase_type_str == "Retreats":
        return
    sorted_all_power_unit_postions = sorted(all_power_unit_postions.items())
    for power_unit_postions, power_order_components, power_user_content, power_possble_orders_list in zip(sorted_all_power_unit_postions, all_power_order_components, user_content, all_power_possble_orders_list):
        genrated_orders = ""

        for power_unit_postion, power_possble_orders in zip(power_unit_postions[1], power_possble_orders_list):
            unit_order = find_unit_order(power_unit_postion, power_order_components)  
            power_possble_order_without_unit = []
            possible_orders_nums = 0
            if power_possble_orders != 'None': 
                for power_possble_order in power_possble_orders:
                    if possible_orders_nums > MAX_POSSIBLE_ORDER_NUM:
                        break
                    power_possble_orders_unit, power_possble_order = get_unit_position_and_order(power_possble_order) 
                    power_possble_order_without_unit.append(power_possble_order)
                    possible_orders_nums += 1
                possble_orders_str_without_unit = ", ".join(power_possble_order_without_unit)
            else:
                power_possble_orders_unit = power_unit_postion
                possble_orders_str_without_unit = "holds"
            assert power_possble_orders_unit.split("'")[0] == power_unit_postion, "可行动作空间和当前单位不匹配."
            # order_component_split = order_component.strip().split("\n")
            
            if genrated_orders == "":
                messages.append({
                    "messages":[
                        {"role": "system", "content": f"{system_content}"},
                        {"role": "user", "content":f"{power_user_content}\n\n[Last Move]:\n{get_last_order_act(last_order_act, power_user_content, year_str, season_str, phase_type_str)}\nIn this round, next is your first order. The candidate orders for {power_unit_postion} are [{possble_orders_str_without_unit}]. The best order from candidate orders is {[power_unit_postion]}"},
                        {"role": "assistant", "content": f"{unit_order}"}
                    ]
                })
                # print(f"{power_user_content}\n\n[Last Move]:\n{get_last_order_act(last_order_act, power_user_content, year_str, season_str, phase_type_str)}\nIn this round, next is your first order. The candidate orders for {power_unit_postion} are [{possble_orders_str_without_unit}]. The best order from candidate orders is {[power_unit_postion]}")
                # print(f"\n{unit_order}\n\n")
                # print(messages[-1])
                genrated_orders += f"{power_unit_postion} {unit_order}"

            else:
                messages.append({
                    "messages":[
                        {"role": "system", "content": f"{system_content}"},
                        {"role": "user", "content":f"{power_user_content}\n\n[Last Move]:\n{get_last_order_act(last_order_act, power_user_content, year_str, season_str, phase_type_str)}\nIn this round, the orders you have previously generated are [{genrated_orders}]. The candidate orders for {power_unit_postion} are [{possble_orders_str_without_unit}]. The best order from candidate orders is that {power_unit_postion}"},
                        {"role": "assistant", "content": f"{unit_order}"}
                    ]
                })
                # print(f"{power_user_content}\n\n[Last Move]:\n{get_last_order_act(last_order_act, power_user_content, year_str, season_str, phase_type_str)}\nIn this round, the orders you have previously generated are [{genrated_orders}]. The candidate orders for {power_unit_postion} are [{possble_orders_str_without_unit}]. The best order from candidate orders is that {power_unit_postion}")
                # print(f"\n{unit_order}\n\n")
                # print(messages[-1])
                genrated_orders += f", {power_unit_postion} {unit_order}" 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="From bin to json")
    parser.add_argument("--input_file_path", type=str, default=INPUT_FILE_PATH)    
    parser.add_argument("--output_file_path", type=str, default=OUTPUT_FILE_PATH)
    parser.add_argument("--num_data", type=int, required=False, default=500)
    parser.add_argument("--num_actions", type=int, default=1)
    parser.add_argument("--playing_countries", nargs="*", default=[], help="List all countries, for example, --playing_countries A B, if its empty or not provided, will default to all countries.")

    args = parser.parse_args()
    assert args.num_actions <= 10, "There are only 10 recommend actions in the dataset, don't pass --num_actions greater than 10."
    if len(args.playing_countries) > 0:
        playing_countries = set(args.playing_countries)

    replay_params = dict(
        seed=1,
        alpha=1.0,
        beta=0.4,
        prefetch=False,
        capacity=10000,
        shuffle=False
    )
    test_buffer = rela.NestPrioritizedReplay(**replay_params)

    print(f"data loading from {args.input_file_path}")
    test_buffer.load(args.input_file_path)
    print(f"data load successfully")

    contents = test_buffer.get_all_content()[0]
    messages = []
    data_count = 0
    print("start transform bin to json")
    for content_dict in tqdm.tqdm(contents, desc="decoding dataset"):
        data_count += 1
        if data_count > MAX_DATA_COUNT:
            break
        board_states = content_dict["observations/x_board_state"]
        years = content_dict["years"]
        seasons = content_dict["observations/x_season"]
        phase_types = content_dict["phase_type"]
        # press = content_dict["observations/x_has_press"]

        search_policy_orders = content_dict["search_policy_orders"]
        search_policy_probs = content_dict["search_policy_probs"]
        possible_actions = content_dict["observations/x_possible_actions"]
        orders = content_dict["orders"]
        last_order_act = ""

        for transition_id in range(board_states.size(0)):
            
            board_state = board_states[transition_id, :]
            year = years[transition_id]
            season = seasons[transition_id, :]
            phase_type = phase_types[transition_id]
            order = orders[transition_id]
            
            description = f"""[Game Time and Phase]
            {year2str(year)} {season2str(season)}: {phase2str(phase_type)}
            [Board State]
            {board_state2state_description_str(board_state)}
            """

            search_policy_order = search_policy_orders[transition_id, :]
            search_policy_prob = search_policy_probs[transition_id, :]
            possible_action = possible_actions[transition_id, :]
            user_content = []
            order_description = []
            curr_order_act = ""
            all_power_order_component_list = []
            all_power_order_components = []
            all_power_possble_orders_list = []
            for country_index in range(7):
                country_name = index2country_str[country_index]

                country_possible_action = possible_action[country_index, :]

                country_search_policy_prob = search_policy_prob[country_index, :]
                country_search_policy_order = search_policy_order[country_index, :]
                
                # system_content = "You are an expert in the board game Diplomacy. Your task is to play as one of seven powers to control over 17 of 34 supply centers. You are observing in [Game Time and Phase] and [Board State]. In [Board State], the regions occupied by each country will be shown in turn, and each region has the format Area(attribute 1, attribute 2, ...). Note that the troops attribute for all regions defaults to (troops non-buildable here, troops non-removable here, no troops dislodged here), which will be omitted in the default state.\n\nNow, generate the action in the correct format, considering the [Board State] and [Game Time and Phase]. Your response should be in the following format:\nThe recommended strategic actions are as follows:\n[Recommend Action ]\n- \n- \n...\n- "
                system_content = "You are an expert in the no-press Diplomacy game environment. As one of seven powers, your task is to use your army and fleet to control the supply center on the board. You are playing [Your Power] and observing [Game Time and Phase], [Board State], and [Last Move] below."
                create_user_content, all_power_unit_postions = create_user_prompt(description, country_name)
                user_content.append(create_user_content)
                full_orders = country_orders2str(country_search_policy_order, country_search_policy_prob, country_possible_action, num_actions=args.num_actions)
                all_power_possble_orders_list.append(country_possible_orders(country_search_policy_order, country_search_policy_prob, country_possible_action, num_actions=50))
                orders_components = full_orders.split("\n")[1:] # 所有动作分量
                # order_component_list = get_action_component(full_orders) # 所有动作增量式分量
                # all_power_order_component_list.append(order_component_list)
                all_power_order_components.append(orders_components) 
                if phase2str(phase_type) == 'Diplomacy':
                    curr_order_act += f"{country_name}:\n{orders2str(order[country_index], country_possible_action)}\n"
                
            get_power_sequence_messages(year2str(year), season2str(season), phase2str(phase_type), all_power_unit_postions, all_power_order_components, user_content, last_order_act, all_power_possble_orders_list)
            if curr_order_act != "":
                last_order_act = curr_order_act
                curr_order_act = ""
    print(f"save message to {OUTPUT_FILE_PATH}")
    with open(OUTPUT_FILE_PATH, "w") as f:
            json.dump(messages, f, indent=2)