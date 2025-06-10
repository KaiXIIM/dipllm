import argparse
import re
import statistics

def extract_score(data):
    # 将数据按行分割
    lines = data.strip().split('\n')
    results = {}
    
    for line in lines:
        # 找到'POWER:'后面的名字，这将是我们要查找得分的国家
        power_name = line.split('POWER:')[1].strip().split(',')[0]
        
        # 找到'Scores:'后面的字典字符串
        scores_str = line.split('Scores:')[1].strip()
        match = re.search(r'\{.*?\}', scores_str)
        if match:
            extracted_dict = match.group(0)
            scores_dict = eval(extracted_dict)
            total_score = 0
            for power, score in scores_dict.items():
                total_score += score ** 2
            sos_score = (scores_dict[power_name] ** 2) / total_score
            if power_name not in results:
                results[power_name] = []
            results[power_name].append(sos_score)
        sorted_results = {key: results[key] for key in sorted(results)}
    
    return sorted_results

playing_countries = {
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY",
    "ITALY", "RUSSIA", "TURKEY"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="From json to json")
    parser.add_argument("--input_file_path", type=str, default='')
    parser.add_argument("--output_file_path", type=str, default='')
    args = parser.parse_args()

    with open(args.input_file_path, 'r') as file:
        print(f"load data from {args.input_file_path}")
        data = file.read()  # 读取文件的全部内容
    all_scores = extract_score(data)
    print(all_scores)
    total_average = 0
    for country, scores in all_scores.items():
        average = round(100*statistics.mean(scores), 2)
        variance = round(100*statistics.variance(scores), 2)
        total_average += average
        print(f"{average}%+{variance}%")
        # print(f"{average}")

        # print(scores)
    overall_average = round(total_average / 7, 2)
    # print(f"All average score: {overall_average}%")
    print(f"{overall_average}%")