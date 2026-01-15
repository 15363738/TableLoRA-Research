import json
import argparse
import re
from tqdm import tqdm
from evaluator import to_value_list, check_denotation

def clean_prediction(pred_str):
    """
    清洗模型输出的“复读机”内容
    """
    if not isinstance(pred_str, str):
        return str(pred_str)

    # 1. 遇到 Llama-3 的特殊结束符直接截断
    # <|eot_id|> 的大概率解码结果，或者直接是字符串形式
    stop_tokens = [
        "<|eot_id|>", 
        "<|end_of_text|>", 
        "<|start_header_id|>", 
        "assistant\n\n",
        "\nUser:", 
        "\n\n"
    ]
    
    for token in stop_tokens:
        if token in pred_str:
            pred_str = pred_str.split(token)[0]

    # 2. 如果包含换行，通常 WikiTQ 的答案很短，只取第一行
    if "\n" in pred_str:
        pred_str = pred_str.split("\n")[0]

    # 3. 处理重复乱码 (如 "Spain overposting overposting...")
    # 简单的启发式：如果一个词重复出现超过 3 次，就从那里截断
    words = pred_str.split()
    if len(words) > 10: # 只有长答案才检查
        for i in range(len(words) - 3):
            # 检查连续三个词是否相同 (A A A)
            if words[i] == words[i+1] == words[i+2]:
                # 截断到这里
                pred_str = " ".join(words[:i])
                break
    
    return pred_str.strip()

def evaluate(pred_file):
    print(f"Loading predictions from {pred_file}...")
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    num_correct = 0
    num_total = 0
    
    # 错误日志
    error_log = []

    for i, line in enumerate(tqdm(lines, desc="Evaluating")):
        item = json.loads(line)
        
        # === 核心修改：清洗预测结果 ===
        raw_prediction = item.get('predict', '')
        prediction_str = clean_prediction(raw_prediction)
        
        predicted_list = [prediction_str] 
        
        label_str = item.get('label', '')
        # 处理 label 中的逗号分隔 (如果预处理时合并了)
        # 注意：有些答案本身包含逗号（如 "10,000"），这里简单 split 可能会误伤
        # 但 WikiTQ 预处理通常是用特殊方式合并的。
        # 如果不确定，可以先假设 label 就是单个字符串
        target_list = [label_str]

        try:
            target_values = to_value_list(target_list)
            predicted_values = to_value_list(predicted_list)
            
            is_correct = check_denotation(target_values, predicted_values)
            
            if is_correct:
                num_correct += 1
            else:
                if len(error_log) < 5:
                    error_log.append({
                        "idx": i,
                        "raw_pred": raw_prediction[:100], # 记录原始的脏数据看看
                        "clean_pred": prediction_str,
                        "gold": label_str,
                    })
                    
            num_total += 1
            
        except Exception as e:
            # print(f"Error processing line {i}: {e}")
            continue

    accuracy = num_correct / num_total * 100 if num_total > 0 else 0
    
    print("\n" + "="*30)
    print(f"Total Examples: {num_total}")
    print(f"Correct: {num_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*30)
    
    if error_log:
        print("\nSample Errors (Top 5):")
        for err in error_log:
            print(f"Gold: {err['gold']}")
            print(f"Pred (Raw)  : {err['raw_pred']}...")
            print(f"Pred (Clean): {err['clean_pred']}")
            print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to generated_predictions.jsonl")
    args = parser.parse_args()
    
    evaluate(args.pred_file)