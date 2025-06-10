# INPUT_FILE_PATH = "/home/xukaixuan/diplomacy_experiments/dataset_with_html/high_datset_0717/c04_exploit/research_20240409_collect_expert_data/lau.loc.use@true_b2bca929/buffer1.bin"
INPUT_FILE_PATH = "/home/xukaixuan/diplomacy_experiments/results/diplomacy/adhoc/2024-07-20T120950.415827/c04_exploit/research_20240409_collect_data/lau.loc.use@true_b2bca929/buffer1.bin"
# OUTPUT_FILE_PATH = "/home/xukaixuan/diplomacy_experiments/Prompt/high_0719/prompt_html_high_full.json"
OUTPUT_FILE_PATH = "/home/xukaixuan/diplomacy_experiments/Prompt/high_0719/prompt_test.json"
from fairdiplomacy.selfplay import rela
import torch
import json
import tqdm
import argparse


from collections import defaultdict


with open("/home/xukaixuan/diplomacy_sft/data_process/order_vocabulary_nl_2.json", "r") as f:
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
    75: "Smyrna", 76: "Syria", 77: "Bulgaria", 78: "Bulgaria's Eastern Coast", 79: "Constantinople",
    80: "Bulgaria's Southern Coast"
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

def create_user_prompt(board_state_content):
    regions = board_state_content.strip().split("\n\n")
    print(regions)
    input()
    occupation_stats = defaultdict(lambda: defaultdict(lambda: {'names': [], 'count': 0, 'attributes': []}))
    first_region = True
    for region in regions:
        region_info = region.split("\n")
    
        if first_region:
            game_time = region_info[1].lstrip().title()
            first_region = False
            region_name = region_info[3].lstrip("[").lstrip().strip("[]")
        else:
            region_name = region_info[0].strip("[]")
        power = None  # 用于存储当前区域的权力归属

        attributes = []
        for line in region_info[0:]:
            if "[Board State]" in line:
                continue
            if "not occupied by" in line:
                power = "Unoccupied areas"
            elif "Occupied by" in line:
                # 确保正确提取占领者名称
                power = line.split("'s ")[0]
                power = ' '.join(line.split()[2:]).rstrip(".")
            # 根据区域描述添加属性:
            if "Can build new troop from here" in line:
                attributes.append("troops buildable")
                # print()
            elif "the troop here can be removed" in line:
                attributes.append("troops removable")
            if "not dislodged troop" not in line and "dislodged" in line:
                attributes.append(line)
            if "this area is" in line and len(line.split()) == 4 and "coast" not in line:
                line_split = line.split()
                attributes.append(line_split[-1].rstrip('.'))
            elif "this area is" in line and len(line.split()) > 5:
                if "home center" in line and "neither" not in line:
                    attributes.append(line[13:])
                elif "this area is a supply center" in line:
                    attributes.append("supply center") 

        if attributes:
            occupation_stats[power][region_name]['attributes'] = ', '.join(attributes)
            occupation_stats[power][region_name]['names'].append(region_name)
            occupation_stats[power][region_name]['count'] += 1

    user_prompt = f"[Game Time and Phase]:\n{game_time}\n\n[Board State]:\n"
    sorted_items = sorted(occupation_stats.items(), key=lambda item: item[0])
    for power, regions in sorted_items:
        user_prompt += f"{power}:"
        sorted_regions = sorted(regions.items(), key=lambda item: item[0])
        for region_name, stats in sorted_regions:
            # 假设 attributes 是一个字符串，如果不是，需要相应地转换
            if stats['attributes'] is not None:
                user_prompt += f"\n- {region_name} ({stats['attributes']})"
            else:
                user_prompt += f"\n- {region_name}"
        user_prompt += "\n"
    return user_prompt.rstrip()

def area_tensor2area_description_str(area_state):
    troop_list = [
        "army", "fleet"
    ]
    country_list = [
        "Austria", "England", "France","Germany",
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
        description += f"{get_dislodged_troop_country(area_state)}'s {get_dislodged_troop(area_state)} was dislodged\n"
    else:
        description += "not dislodged troop here\n"
    description += f"this area is {get_area_type(area_state)}.\n"
    if get_home_center_country(area_state) is not None:
        description += f"this area is {get_home_center_country(area_state)}'s home center.\n"
    elif get_supply_center_country(area_state) is not None:
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
    order_idxs = []
    for idx in range(orders.size(0)):
        order_idxs.append(possible_action[idx][orders[idx]])
    orders_str = ""
    for order_idx in order_idxs:
        if int(order_idx) != -1:
            orders_str += f"- {index2order[str(int(order_idx))]}\n"
    orders_str = orders_str.replace(";", "\n")
    # print(orders_str)
    # input()
    return orders_str

def single_orders2str(orders, possible_action):
    order_idxs = []
    for idx in range(orders.size(0)):
        order_idxs.append(possible_action[idx][orders[idx]])
    orders_str = ""
    for order_idx in order_idxs:
        if int(order_idx) != -1:
            orders_str += f"- {index2order[str(int(order_idx))]}\n"
    orders_str = orders_str.replace(";", "\n")
    # print(orders_str)
    # input()
    return orders_str

def country_orders2str(country_search_policy_order, country_search_policy_prob, country_possible_action, num_actions=1):
    combined_order_str = "The recommended strategic actions are as follows:\n"
    largest_prob_indices = torch.topk(country_search_policy_prob, k=num_actions, largest=True).indices.long()
    count = 1
    for idx in largest_prob_indices: 
        combined_order_str += f"[Recommend Action {count}]\n"
        order = country_search_policy_order[idx, :]
        combined_order_str += orders2str(order, country_possible_action)
        combined_order_str += "\n"
        count += 1
    return combined_order_str.rstrip()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="From bin to json")
    parser.add_argument("--input_file_path", type=str, default=INPUT_FILE_PATH)    
    parser.add_argument("--output_file_path", type=str, default=OUTPUT_FILE_PATH)
    parser.add_argument("--num_data", type=int, required=False, default=150)
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
    print("start transform bin to json")
    data_count = 0
    for content_dict in tqdm.tqdm(contents, desc="decoding dataset"):
        board_states = content_dict["observations/x_board_state"]
        years = content_dict["years"]
        seasons = content_dict["observations/x_season"]
        phase_types = content_dict["phase_type"]

        search_policy_orders = content_dict["search_policy_orders"]
        search_policy_probs = content_dict["search_policy_probs"]
        possible_actions = content_dict["observations/x_possible_actions"]

        next_board_states = content_dict["next_observations/x_board_state"]
        # next_years = content_dict["next_years"]
        next_seasons = content_dict["next_observations/x_season"]
        # next_phase_types = content_dict["next_phase_type"]

        orders = content_dict["orders"]

        # print(orders)
        # input()
        for transition_id in range(board_states.size(0)-1):
            occupation_stats = defaultdict(lambda: defaultdict(lambda: {'powers': [], 'count': 0, 'attributes': []}))
            board_state = board_states[transition_id, :]
            year = years[transition_id]
            season = seasons[transition_id, :]
            phase_type = phase_types[transition_id]
            order = orders[transition_id]
            next_board_state = next_board_states[transition_id, :]
            next_year = years[transition_id+1]
            next_phase_type = phase_types[transition_id+1]

            next_season = next_seasons[transition_id, :]
            # next_phase_type = next_phase_types[transition_id]
            
            
            description = f"""[Game Time and Phase]
            {year2str(year)} {season2str(season)}: {phase2str(phase_type)}
            [Board State]
            {board_state2state_description_str(board_state)}
            """

            next_description = f"""\n[Game Time and Phase]
            {year2str(next_year)} {season2str(next_season)}: {phase2str(next_phase_type)}
            [Board State]
            {board_state2state_description_str(next_board_state)}\n
            """
            # next_description = f"""[Next State]\n{next_description}"""

            search_policy_order = search_policy_orders[transition_id, :]
            search_policy_prob = search_policy_probs[transition_id, :]
            possible_action = possible_actions[transition_id, :]

            if data_count >= args.num_data:
                break 
            data_count += 1

            order_act = ""
            for country_index in range(7):
                country_name = index2country_str[country_index]
                

                country_possible_action = possible_action[country_index, :]
                country_search_policy_prob = search_policy_prob[country_index, :]
                country_search_policy_order = search_policy_order[country_index, :]
                
                # order_description = country_orders2str(country_search_policy_order, country_search_policy_prob, country_possible_action, num_actions=args.num_actions)
                order_possible_action =  torch.ones(country_possible_action.shape)
                order_act += f"[{country_name}]:\n{orders2str(order[country_index], country_possible_action)}"
                # system_content = "You are an expert in the board game Diplomacy. Your task is to play as one of seven powers to control over 17 of 34 supply centers. You are observing in [Game Time and Phase] and [Board State]. In [Board State], the regions occupied by each country will be shown in turn, and each region has the format Area(attribute 1, attribute 2, ...). Note that the troops attribute for all regions defaults to (troops non-buildable here, troops non-removable here, no troops dislodged here), which will be omitted in the default state.\n\nNow, generate the action in the correct format, considering the [Board State] and [Game Time and Phase]. Your response should be in the following format:\nThe recommended strategic actions are as follows:\n[Recommend Action ]\n- \n- \n...\n- "
            system_content = "You are a world model in the no-press Diplomacy environment. Your task is to predict the [NEXT STATE] based on the [CURRENT STATE] and [ACTION]. [CURRENT STATE] contains [Game Time and Phase] and [Board State], where each power will sequentially display its occupied areas, each with the format Area (attribute 1, attribute 2,...). Remember, the default attributes for areas—(coast, neither supply center nor home center, troops non-buildable here, troops non-removable here, no troops dislodged here)—will be omitted unless specified otherwise."
            user_content = f"[CURRENT STATE]:\n{create_user_prompt(description)}\n\n[ACTION]:\n{order_act}"
            order_description = f"[NEXT STATE]:\n{create_user_prompt(next_description)}"
            # print(system_content)
            # input()
            # print(user_content)
            # input()
            # print(order_description)
            # input()
            messages.append({
                "messages":[
                    {"role": "system", "content": f"{system_content}"},
                    {"role": "user", "content": f"{user_content}"},
                    {"role": "assistant", "content": f"{order_description}"}
                ]
            })
        

    print(f"save message to {OUTPUT_FILE_PATH}")
    with open(OUTPUT_FILE_PATH, "+w") as f:
            json.dump(messages, f, indent=2)

