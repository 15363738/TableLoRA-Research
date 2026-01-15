# Copyright 2024 the LlamaFactory team.
# Adapted for TableLoRA support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from copy import deepcopy
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Tuple[List[int], List[int]]:
    if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    messages = prompt + response
    input_ids, labels = [], []

    if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids += [image_token_id] * getattr(processor, "image_seq_length")
        labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = 1 if template.efficient_eos else 0
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= data_args.cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), data_args.cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if data_args.train_on_prompt:
            source_label = source_ids
        elif turn_idx != 0 and template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if data_args.mask_history and turn_idx != len(encoded_pairs) - 1:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        input_ids += source_ids + target_ids
        labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    
    # === TableLoRA Init Start ===
    # 检查是否开启了 TableLoRA，并初始化额外的 row_ids/col_ids 字段
    if getattr(data_args, "emb_lora", False):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "row_ids": [], "col_ids": []}
        ori_tokenizer = deepcopy(tokenizer)
        # 添加特殊的 <TAB> token
        if "<TAB>" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["<TAB>"], special_tokens=True)
    else:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    # === TableLoRA Init End ===

    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        # === TableLoRA Logic: Extract Table Start ===
        current_prompt = examples["prompt"][i]
        table_texts = []
        
        if getattr(data_args, "emb_lora", False):
            # 注意：TableLoRA 假设 Prompt 是单轮或表格在第一轮
            prompt_content = current_prompt[0]['content']
            pattern = r"/\*\n(.*?)\n\*/"  
            
            matches = re.findall(pattern, prompt_content, re.DOTALL)
            if matches:
                # 将表格内容替换为 <TAB> 占位符
                replaced_text = re.sub(pattern, "/*\n<TAB>\n*/", prompt_content, flags=re.DOTALL)
                # 使用深拷贝避免修改原始数据
                current_prompt = deepcopy(current_prompt)
                current_prompt[0]['content'] = replaced_text
                table_texts = matches
        # === TableLoRA Logic: Extract Table End ===

        # 调用 v0.8.3 原生的编码函数 (注意：传参方式必须符合 v0.8.3 的签名)
        input_ids, labels = _encode_supervised_example(
            prompt=current_prompt,
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args, # 这里直接传对象
        )
        
        # === TableLoRA Logic: Process IDs Start ===
        if not getattr(data_args, "emb_lora", False) or len(table_texts) == 0:
            # 普通数据处理逻辑
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if getattr(data_args, "emb_lora", False):
                model_inputs["row_ids"].append([0] * len(input_ids))
                model_inputs["col_ids"].append([0] * len(input_ids))
        
        else:
            # TableLoRA 特殊处理逻辑
            # 判断表格格式是带有 <ROW>/<COL> 标签的还是 Markdown
            is_tagged_format = "<ROW>" in table_texts[0] and "<COL>" in table_texts[0]
            
            row_ids_total = [0] * len(input_ids)
            col_ids_total = [0] * len(input_ids)
            
            for table_text in table_texts:
                table_token_ids = []
                col_ids = []
                row_ids = []
                
                if is_tagged_format:
                    # 处理带标签的格式 <ROW>...<COL>...
                    # Header
                    header_part = table_text.split("<ROW>")[0]
                    ids = tokenizer.encode(header_part, add_special_tokens=False) # 避免首部添加 BOS
                    table_token_ids.extend(ids)
                    row_ids.extend([0] * len(ids))
                    col_ids.extend([0] * len(ids))
                    
                    row_id = 1
                    for row_text in table_text.split("<ROW>")[1:]:
                        col_id = 0
                        row_split = row_text.split("<COL>")
                        # Row header / first cell
                        ids = tokenizer.encode("<ROW>" + row_split[0], add_special_tokens=False)
                        table_token_ids.extend(ids)
                        row_ids.extend([row_id] * len(ids))
                        col_ids.extend([0] * len(ids))
                        col_id += 1
                        
                        for col_text in row_split[1:]:
                            ids = tokenizer.encode("<COL>" + col_text, add_special_tokens=False)
                            table_token_ids.extend(ids)
                            row_ids.extend([row_id] * len(ids))
                            col_ids.extend([col_id] * len(ids))
                            col_id += 1
                        row_id += 1
                else:
                    # 处理 Markdown 格式
                    row_id = 1
                    for row_text in table_text.split("\n"):
                        # 跳过分隔符行 |---|
                        if set(row_text.strip()) <= set(["|", "-", ":", " "]):
                            # TableLoRA 逻辑：如果是分隔行，回退 row_id 还是保持？
                            # 原代码逻辑比较简单，我们这里简化处理：
                            # 把它当作普通文本编码，row_id 保持上一个（即0或header）
                            # 但原代码有一个 `row_id -= 1` 然后编码再 `row_id += 1` 的操作，也就是把它归为上一行
                            row_id -= 1
                            ids = tokenizer.encode(row_text + "\n", add_special_tokens=False)
                            table_token_ids.extend(ids)
                            row_ids.extend([row_id] * len(ids))
                            col_ids.extend([0] * len(ids))
                            row_id += 1
                        else:
                            # 处理普通行
                            parts = row_text.split("|")
                            # 重新构造行首： | cell1 |
                            # 这里简单处理，原代码逻辑是对 split 后的部分分别编码
                            # 前两部分（行首和第一列前）
                            prefix = "|" + "|".join(parts[:2]) + "|" if len(parts) > 2 else row_text
                            ids = tokenizer.encode(prefix, add_special_tokens=False)
                            table_token_ids.extend(ids)
                            row_ids.extend([row_id] * len(ids))
                            col_ids.extend([0] * len(ids))
                            col_id = 1 # 假设第一部分是第0列，这里开始第1列
                            
                            for col_text in parts[2:]:
                                ids = tokenizer.encode(col_text + "|", add_special_tokens=False)
                                table_token_ids.extend(ids)
                                row_ids.extend([row_id] * len(ids))
                                col_ids.extend([col_id] * len(ids)) # 简化：列ID统一递增，不区分细致
                                col_id += 1
                            row_id += 1

                # 将生成的 IDs 插入到 input_ids 中 <TAB> 的位置
                tab_token_id = tokenizer.convert_tokens_to_ids("<TAB>")
                if input_ids.count(tab_token_id) == 0:
                    break
                
                table_insert_idx = input_ids.index(tab_token_id)
                
                # 拼接 IDs
                row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
                col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
                
                # 拼接 Label (根据是否 train_on_prompt 决定是否 mask 表格)
                if data_args.train_on_prompt:
                    labels = labels[:table_insert_idx] + table_token_ids + labels[table_insert_idx+1:]
                else:
                    labels = labels[:table_insert_idx] + [IGNORE_INDEX] * len(table_token_ids) + labels[table_insert_idx+1:]
                
                # 拼接 Input IDs
                input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]

            model_inputs["row_ids"].append(row_ids_total)
            model_inputs["col_ids"].append(col_ids_total)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            
            # 简单校验
            if "<TAB>" in tokenizer.decode(input_ids):
                 logger.warning("Example contains unreplaced <TAB> token.")
        # === TableLoRA Logic: Process IDs End ===

        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))

    # 还原 tokenizer 状态
    if getattr(data_args, "emb_lora", False):
        tokenizer = ori_tokenizer
        
    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    valid_num = 0
    batch_input_ids, batch_labels = [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=None,
            data_args=data_args,
        )
        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning("Dropped lengthy example with length {} > {}.".format(length, data_args.cutoff_len))
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            valid_num += 1

    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len)
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))