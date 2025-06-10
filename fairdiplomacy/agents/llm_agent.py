#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



import logging
from typing import Dict, Optional, Sequence
import random
import json
import os
import torch
import itertools
import requests
import torch.cuda
import time

from conf import agents_cfgs
from data_process import bin2json_0903 as dataprocess
from fairdiplomacy.agents.base_agent import AgentState, BaseAgent, NoAgentState
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power
from fairdiplomacy.utils.parse_device import device_id_to_str
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from fairdiplomacy.utils.translator import (create_user_prompt_hier, index2country_str, board_state2state_description_str,
                                            year2str, season2str, phase2str, country_str2index)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import util, SentenceTransformer

playing_countries = [
    "Austria", "England", "France", "Germany",
    "Italy", "Russia", "Turkey"
]
MAX_ACTION_SPACE = 5

class LLMAgent(BaseAgent):
    def __init__(self, cfg: agents_cfgs.BaseStrategyModelAgent):
        self.device = device_id_to_str(cfg.device)
        self.temperature = 0.1
        self.top_p = 0.95
        self.has_press = cfg.has_press
        self.possible_actions_num = 1
        self.global_prev_orders_path = None
        self.port = cfg.port
        self.out_of_possible_orders_count = 0
        # self.blueprint_model = BaseStrategyModelWrapper(
        #     'models/no_press_human_imitation_policy.ckpt', device=self.device, half_precision=cfg.half_precision
        # )
        self.blueprint_model = BaseStrategyModelWrapper(
            'models/diplodocus_low_rl_policy.ckpt', device=self.device, half_precision=cfg.half_precision
        )
        if not torch.cuda.is_available() and self.device != "cpu":
            logging.warning("Using cpu because cuda not available")
            self.device = "cpu"
        self.file_path = 'llm.txt'
        self.input_encoder = FeatureEncoder()
        with open('data_process/new_order_vocabulary_nl_2.json') as f:
            self.rich_orders = json.load(f)
        
        with open('data_process/new_order_vocabulary.json', 'r') as f:
            self.original_orders = json.load(f)

        self.index_to_order_index = {index: order for index, order in enumerate(self.rich_orders)}
        self.llm_topk = 10

        self.max_attempts = 10
        self.max_sequence_length = 30
        self.order_path = 'diplomacy_experiments/prompt_orders.json'

    def order_to_language_order(self, original_order):
        index = -1
        for key, value in self.original_orders.items():
            if value == original_order:
                index = key
                break
        return self.rich_orders[index]
    
    def language_order_to_order(self, language_order):
        index = -1
        for key, value in self.rich_orders.items():
            if value == language_order:
                index = key
                break
        if index != -1:
            return self.original_orders[index]
        else:
            return False
    
    def llm_response(self, prompt_data):
        url = "http://172.18.36.111:{}/response".format(self.port)
        data = {
            "prompt_data": prompt_data
        }
        response = requests.post(url, json=data)
        response_str = response.json()['response']

        return response_str

    def country_possible_orders(self, country_possible_orders_list):
        sorted_country_possible_orders_list = sorted(country_possible_orders_list, key=lambda x: x['prob'], reverse=True)[0:self.llm_topk]

        return [sorted(item['actions']) for item in sorted_country_possible_orders_list]

    
    def get_orders(self, game: Game, power: Power, state: AgentState):
        assert isinstance(state, NoAgentState)
        if len(game.get_orderable_locations().get(power, [])) == 0:
            return tuple()
        return self._get_orders_many_powers(game, [power], agent_power=power)[power]
    
    def get_orders(self, game: Game, power: Power, state: AgentState):
        assert isinstance(state, NoAgentState)
        if len(game.get_orderable_locations().get(power, [])) == 0:
            return tuple()
        return self._get_orders_many_powers(game, [power], agent_power=power)[power]

    def get_orders_many_powers(self, game: Game, powers: Sequence[Power]):
        return self._get_orders_many_powers(game, powers, agent_power=None)

    def _get_orders_many_powers(
            self, game: Game, powers: Sequence[Power], agent_power: Optional[Power]):
        # all_possible_orders = game.get_all_possible_orders()
        all_possible_orders = {
            loc: orders
            for loc, orders in game.get_all_possible_orders().items()
            if loc in game.get_orderable_locations()[powers[0]]
        }
        all_possible_orders = set(itertools.chain.from_iterable(all_possible_orders.values()))
        orders = {}

        for power in powers:
            sampled_orders = self.get_sampled_actions_for_power(
                blueprint_model=self.blueprint_model,
                game=game,
                power=power,
                powers=powers,
                playing_countries=playing_countries,
                has_press=self.has_press,
                temperature=self.temperature,
                top_p=self.top_p,
                possible_actions_num=self.possible_actions_num,
                order_to_language_order_fn=self.order_to_language_order,
            )
            unit_possible_orders_list = self.country_possible_orders(sampled_orders)
            sequecnce_length = 0
            cur_order_list = []
            cur_language_order = []
            system_content, user_content = self.translate_game_to_turn_prompt(game, power)
            for unit_index in range(len(unit_possible_orders_list[0])):
                attempts = 0
                prompt, unit_position, unit_possible_order = self.translate_game_to_prompt(system_content, user_content, cur_language_order, unit_possible_orders_list, unit_index, cur_language_order)
                action_spcace = len(unit_possible_order)
                if action_spcace == 1:
                    cur_order_list.append(self.language_order_to_order(f"{unit_position} {unit_possible_order[0]}"))
                    cur_language_order.append(f"{unit_position} {unit_possible_order[0]}")
                    self.valid_action_count += 1
                    attempts = 1
                    continue
                order_backup = unit_possible_order[0]

                while attempts < self.max_attempts:
                    response = self.llm_response(prompt)
                    self.response_count += 1
                    response = f"{unit_position} {response}"

                    logging.info(f"response: {response}")
                    attempts += 1
                    
                    cur_order_language = response.strip('[').strip(']')
                    cur_order = self.language_order_to_order(cur_order_language)
                    
                    if cur_order in all_possible_orders:
                        cur_order_list.append(cur_order)
                        cur_language_order.append(cur_order_language)

                        sequecnce_length += 1
                        self.valid_action_count += 1
                        break
                    else:
                        logging.info(f"invalid action {cur_order}")
                        with open(self.order_path, 'a') as file:
                            file.write(f"{prompt}\n\ninvalid orders:\n{response}\n{cur_order}\n\n\n")

                    if attempts == self.max_attempts:
                        self.invalid_action_count+=1
                        order_backup_standard = self.language_order_to_order(order_backup)
                        cur_order_list.append(order_backup_standard)
                        cur_language_order.append(order_backup)
                        logging.info(f"Replace output {power}'s action:\n {response} by {order_backup} after {self.max_attempts} attempts")
                
            if len(cur_order_list) > 0:
                orders[power] = tuple(cur_order_list)
            else:
                orders[power] = tuple('')

        os.remove(self.global_prev_orders_path)

        return orders

    def can_share_strategy(self) -> bool:
        return True

    def translate_response_to_order(self, response):
        response = response.strip('[').strip(']').strip()
        encoded_input = self.sentence_tokenizer([response], padding=True, truncation=True, return_tensors='pt').to(self.device)
        # model_output = torch.tensor(self.sentence_model.encode([response])).to(self.device)
        with torch.no_grad():
            model_output = self.sentence_model(**encoded_input)[1][0]

        cosine_scores = util.pytorch_cos_sim(model_output, self.rich_embeddings)
        most_similar_index = cosine_scores.argmax().item()
        most_similar_order_index = self.index_to_order_index[most_similar_index]
        most_similar_order = self.original_orders[most_similar_order_index]

        return most_similar_order
    
    def orders2str(self, orders, possible_action):
        order_idxs = set()  # 使用集合确保索引唯一
        for idx in range(orders.size(0)):
            order_idx = possible_action[idx][orders[idx]]
            if int(order_idx) != -1:
                order_idxs.add(order_idx)
        
        orders_strs = [f"- {self.rich_orders[str(int(order_idx))]}\n" for order_idx in order_idxs]  # 生成订单字符串列表
        sorted_orders_strs = sorted(orders_strs)  # 对订单字符串进行排序
        
        orders_str = "".join(sorted_orders_strs)  # 连接字符串
        if len(orders_str) == 0:
            orders_str = "None\n"
        orders_str = orders_str.replace(";", "\n")
        return orders_str  # 移除尾部的换行符

    def get_last_order_act_str(self, prev_power_orders, your_power, year, season, phase_type):
        if year == 1901 and season[0] == torch.tensor(1.) and phase_type == 77: # 1901 Spring Diplomacy
            return "None"
        else:
            language_order = ""
            # prev_power_orders = sorted(prev_power_orders.keys(), key=lambda s: s.title())
            for power, orders in prev_power_orders.items():
                if power.title() == your_power:
                    language_order += f"Your Power Order:\n{power.capitalize()}:\n"
                    if len(orders) == 0:
                        language_order += "None\n"
                    else:
                        for order in orders:
                            language_order += f"{self.get_language_previous_generate_order(order)}\n"
                    break
            language_order += "Other Power Order:\n"
            for power, orders in prev_power_orders.items():
                if power.title() == your_power:
                    continue
                language_order += f"{power.capitalize()}:\n"
                if len(orders) == 0:
                    language_order += "None\n"
                else:
                    for order in orders:
                        language_order += f"{self.get_language_previous_generate_order(order)}\n"
                    
        return language_order.strip('\n')  # 移除最后的换行符

    def get_language_previous_generate_order(self, previous_generate_order):
        language_previous_generate_order = ''
        is_index_exist = False
        for key, value in self.original_orders.items():
            if value == previous_generate_order:
                index_of_cur_order = int(key)  # 将找到的键转换为整数
                is_index_exist = True
        
        if is_index_exist:
            language_previous_generate_order += f"{self.rich_orders[str(index_of_cur_order)]}"
        return language_previous_generate_order
    
    def get_previous_generate_order_str(self, previous_generate_orders, power_unit_postion, possble_orders_str_without_unit):
        if len(previous_generate_orders) == 0:
            return f"In this round, next is your first order. The candidate orders for {power_unit_postion} are [{possble_orders_str_without_unit}]. The best order from candidate orders is that {power_unit_postion}"
        else:
            language_previous_generate_order = ''
            for previous_generate_order in previous_generate_orders:
                language_previous_generate_order += f"{previous_generate_order}, "
            language_previous_generate_order = language_previous_generate_order.strip(', ').strip(' ')
            return f"In this round, the orders you have previously generated are [{language_previous_generate_order}]. The candidate orders for {power_unit_postion} are [{possble_orders_str_without_unit}]. The best order from candidate orders is that {power_unit_postion}"

    def get_possible_orders_space_str(self, all_unit_possible_orders_list, unit_index, previous_generate_order):
        unit_order_without_unit_set = set() 
        for all_unit_orders in all_unit_possible_orders_list:
            _, unit_order_without_unit = dataprocess.get_unit_position_and_order(all_unit_orders[unit_index])
            if all_unit_orders[0:unit_index] == previous_generate_order:
                unit_order_without_unit_set.add(unit_order_without_unit)
        if unit_order_without_unit_set:
            return list(unit_order_without_unit_set)
        else:
            for all_unit_orders in all_unit_possible_orders_list:
                _, unit_order_without_unit = dataprocess.get_unit_position_and_order(all_unit_orders[unit_index])
                unit_order_without_unit_set.add(unit_order_without_unit)
            self.out_of_possible_orders_count += 1
            logging.info(f"maybe last order is not in the possible orders, so we use all possible orders: {unit_order_without_unit_set}")
            return list(unit_order_without_unit_set)

    def translate_game_to_turn_prompt(self, game, power):
        observation = [self.input_encoder.encode_inputs, self.input_encoder.encode_inputs_all_powers][
                    False]([game], input_version=3)
        board_states = observation["x_board_state"]
        year = game.current_year
        seasons = observation["x_season"]
        phase_type = ord(game.current_short_phase[-1])
        board_state = board_states[0, :]
        season = seasons[0, :]
        description = f"""[Game Time and Phase]
            {year2str(year)} {season2str(season)}: {phase2str(phase_type)}
            [Board State]
            {board_state2state_description_str(board_state)}
            """
        system_content = "You are an expert in the no-press Diplomacy game environment. As one of seven powers, your task is to use your army and fleet to control the supply center on the board. You are playing [Your Power] and observing [Game Time and Phase], [Board State], and [Last Move] below."
        user_content, _ = dataprocess.create_user_prompt(description, power)
        self.global_prev_orders_path = f'global_prev_order/{power.upper()}/{game.game_id}.json'

        with open(self.global_prev_orders_path, 'r') as file:
            global_prev_orders = json.load(file)

        last_order_act = self.get_last_order_act_str(global_prev_orders, power, year, season, phase_type)

        user_content = f"{user_content}\n\n[Last Move]:\n{last_order_act}"
        # user_content =f"{user_content}\n\n[Last Move]:\n{last_order_act}\nIn this round, next is your first order. The candidate orders for {unit_position} are [{possible_orders_str}]. The best order from candidate orders is {[unit_position]}"},

        # print(user_content)
        # logging.info(f"user_content: {user_content}")

        return system_content, user_content
    
    def translate_game_to_prompt(self, system_content, user_content, previous_generate_order, all_unit_possible_orders_list, unit_index, langauge_previous_generate_order):

        unit_position, _ = dataprocess.get_unit_position_and_order(all_unit_possible_orders_list[0][unit_index])
        possible_orders_list = self.get_possible_orders_space_str(all_unit_possible_orders_list, unit_index, langauge_previous_generate_order)
        possible_orders_str = ", ".join(possible_orders_list)
        previous_generate_order_str = self.get_previous_generate_order_str(previous_generate_order, unit_position, possible_orders_str)

        user_content = f"{user_content}\n\n{previous_generate_order_str}"
        message = [
            {"role": "system", "content": f"{system_content}"},
            {"role": "user", "content": f"{user_content}"}
        ]

        return message, unit_position, possible_orders_list
    
    def get_sampled_actions_for_power(
        blueprint_model,
        game,
        power,
        powers,
        playing_countries,
        has_press,
        temperature,
        top_p,
        possible_actions_num,
        order_to_language_order_fn
    ):
        power_index = playing_countries.index(power.title())
        all_unit_possible_orders_list = []
        exist_action_list = []
        possible_orders_num = 0

        while possible_orders_num < possible_actions_num:
            actions, _logprobs = blueprint_model.forward_policy(
                [game],
                has_press=has_press,
                agent_power=power,
                temperature=temperature,
                top_p=top_p,
            )

            sampled_actions = actions[0][power_index]

            if sampled_actions not in exist_action_list:
                exist_action_list.append(sampled_actions)
                language_orders = [order_to_language_order_fn(a) for a in sampled_actions]
                all_unit_possible_orders_list.append(language_orders)

            possible_orders_num += 1

        return all_unit_possible_orders_list
