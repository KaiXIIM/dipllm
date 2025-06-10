import json
import re

unit_type_to_NL = {
    "A": "army",
    "F": "fleet",
    "*A": "lost army",
    "*F": "lost fleet"
}

def unit_to_natural_language(unit):
    unit = unit.strip()
    return unit_type_to_NL[unit]

def unit_and_position_to_natural_language(unit_and_position):
    # convert "A/F pos" to "ARMY/FLEET in pos"
    unit_and_position = unit_and_position.strip()
    match = re.match("(\*?[AF]) ([\w/]+)", unit_and_position)
    return f"{unit_to_natural_language(match.group(1))} in {match.group(2)}"

def order_to_natural_language(order):
    order = order.strip()
    if re.match("^(\*?[AF] [\w/]+) H$", order) is not None:
        # A/F pos H
        match = re.match("^(\*?[AF] [\w/]+) H$", order)
        unit_and_position = match.group(1)
        return f"{unit_and_position_to_natural_language(unit_and_position)} hold"
    elif re.match("^(\*?[AF] [\w/]+) R ([\w/]+)$", order) is not None:
        # A/F pos R pos
        match = re.match("^(\*?[AF] [\w/]+) R ([\w/]+)$", order)
        unit_and_position = match.group(1)
        target_position = match.group(2)
        return f"{unit_and_position_to_natural_language(unit_and_position)} retreat to {target_position}"
    elif re.match("^(\*?[AF] [\w/]+) - ([\w/]+)$", order) is not None:
        # A/F pos - pos
        match = re.match("^(\*?[AF] [\w/]+) - ([\w/]+)$", order)
        unit_and_position = match.group(1)
        target_position = match.group(2)
        return f"{unit_and_position_to_natural_language(unit_and_position)} move to {target_position}"
    elif re.match("^(\*?[AF] [\w/]+) - ([\w/]+) VIA$", order) is not None:
        # A/F pos - pos VIA
        match = re.match("^(\*?[AF] [\w/]+) - ([\w/]+) VIA$", order)
        unit_and_position = match.group(1)
        target_position = match.group(2)
        return f"{unit_and_position_to_natural_language(unit_and_position)} move to {target_position} via FLEET"
    elif re.match("^(\*?[AF] [\w/]+) S (\*?[AF] [\w/]+)$", order) is not None:
        # A/F pos S A/F pos
        match = re.match("^(\*?[AF] [\w/]+) S (\*?[AF] [\w/]+)$", order)
        unit_and_position_1 = match.group(1)
        unit_and_position_2 = match.group(2)
        return f"{unit_and_position_to_natural_language(unit_and_position_1)} support {unit_and_position_to_natural_language(unit_and_position_2)}"
    elif re.match("^(\*?[AF] [\w/]+) S (\*?[AF] [\w/]+) - ([\w/]+)$", order) is not None:
        # A/F pos S A/F pos - pos
        match = re.match("^(\*?[AF] [\w/]+) S (\*?[AF] [\w/]+) - ([\w/]+)$", order)
        unit_and_position_1 = match.group(1)
        unit_and_position_2 = match.group(2)
        target_position = match.group(3)
        return f"{unit_and_position_to_natural_language(unit_and_position_1)} support {unit_and_position_to_natural_language(unit_and_position_2)} move to {target_position}"
    elif re.match("^(\*?[AF] [\w/]+) C (\*?[AF] [\w/]+) - ([\w/]+)$", order) is not None:
        # A/F pos C A/F pos - pos
        match = re.match("^(\*?[AF] [\w/]+) C (\*?[AF] [\w/]+) - ([\w/]+)$", order)
        unit_and_position_1 = match.group(1)
        unit_and_position_2 = match.group(2)
        target_position = match.group(3)
        return f"{unit_and_position_to_natural_language(unit_and_position_1)} convoy {unit_and_position_to_natural_language(unit_and_position_2)} to {target_position}"
    elif re.match("^(\*?[AF] [\w/]+) B$", order) is not None:
        # A/F pos B
        match = re.match("^(\*?[AF] [\w/]+) B$", order)
        unit_and_position = match.group(1)
        return f"build {unit_and_position_to_natural_language(unit_and_position)}"
    elif re.match("^(\*?[AF] [\w/]+) D$", order) is not None:
        # A/F pos D
        match = re.match("^(\*?[AF] [\w/]+) D$", order)
        unit_and_position = match.group(1)
        return f"disband {unit_and_position_to_natural_language(unit_and_position)}"
    elif order == "":
        return ""
    else:
        return order


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

area_short2area_full = {
    value.replace("_", "/"): index2area_str_full[key] for key, value in index2area_str.items()
}

with open("/home/xukaixuan/diplomacy_sft/data_process/order_vocabulary.json", "r") as f:
    t = json.load(f)

def orders_to_nl(orders):
    result = [order_to_natural_language(order) for order in orders.split(";")]
    result = ";".join(result)
    return result

tt = {
    key: orders_to_nl(value)
    for key, value in t.items()
}

t = tt

tt = {}
for idx, action in t.items():
    for area_short in area_short2area_full.keys():
        match = re.match(f"(.*) {area_short}(?![/\w])(.*)", action)
        if match is not None:
            action = f"{match.group(1)} {area_short2area_full[area_short]}{match.group(2)}"
    tt[idx] = action

with open("./order_vocabulary_nl_2.json", "a") as f:
    json.dump(tt, f, indent=2)