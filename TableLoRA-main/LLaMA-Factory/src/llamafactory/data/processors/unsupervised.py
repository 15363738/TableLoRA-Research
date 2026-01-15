# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
import re
from copy import deepcopy
from ...extras.logging import get_logger
from ..data_utils import Role
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def _encode_unsupervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    data_args: "DataArguments",
) -> Tuple[List[int], List[int]]:
    if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    if len(response) == 1:
        messages = prompt + response
    else:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids = [image_token_id] * getattr(processor, "image_seq_length") + input_ids

    source_len, target_len = infer_seqlen(len(input_ids), len(labels), data_args.cutoff_len)
    input_ids = input_ids[:source_len]
    labels = labels[:target_len]
    return input_ids, labels


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    if getattr(data_args, "emb_lora", False):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "row_ids": [], "col_ids": []}
        ori_tokenizer = deepcopy(tokenizer)
        if "<TAB>" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["<TAB>"], special_tokens=True)
    else:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        # === TableLoRA Logic Start ===
        table_texts = []
        if getattr(data_args, "emb_lora", False):
            if len(examples["prompt"][i]) != 1:
                # TableLoRA expects single turn prompt for extraction
                # You might want to log a warning instead of raising error to be robust
                pass 
                # raise ValueError(f'prompt should have only one turn')
            
            pattern = r"/\*\n(.*?)\n\*/"
            # Function to replace matched pattern and extract "table"
            def replace_and_extract(text):
                matches = re.findall(pattern, text, re.DOTALL)
                replaced_text = re.sub(pattern, "/*\n<TAB>\n*/", text, flags=re.DOTALL)
                return replaced_text, matches
            
            # Use deepcopy/modify logically to avoid side effects
            prompt_content = examples["prompt"][i][0]['content']
            new_content, table_texts = replace_and_extract(prompt_content)
            
            # Temporary modify the prompt content for encoding
            # We construct a new prompt object to pass to the encoder
            current_prompt = deepcopy(examples["prompt"][i])
            current_prompt[0]['content'] = new_content
        else:
            current_prompt = examples["prompt"][i]
        # === TableLoRA Logic End ===

        input_ids, labels = _encode_unsupervised_example(
            prompt=current_prompt,
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            data_args=data_args,
        )

        # === TableLoRA Injection Start ===
        if not getattr(data_args, "emb_lora", False) or len(table_texts) == 0:
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if getattr(data_args, "emb_lora", False):
                model_inputs["row_ids"].append([0] * len(input_ids))
                model_inputs["col_ids"].append([0] * len(input_ids))
        
        elif "<ROW>" in table_texts[0] and "<COL>" in table_texts[0]:
            row_ids_total = [0] * len(input_ids)
            col_ids_total = [0] * len(input_ids)
            for table_text in table_texts:
                # Calculate row ids and col ids
                table_token_ids = []
                col_ids = []
                row_ids = []
                
                # Header part
                # Note: TableLoRA original code slices [1:] (tokenizer.encode(...)[1:])
                # This assumes encode adds a BOS token. LLaMA tokenizer usually does. 
                # If using add_special_tokens=False, we should NOT slice [1:].
                # For safety/consistency with supervised.py, we use add_special_tokens=False logic if possible,
                # BUT the original TableLoRA code used simple encode() which implies special tokens.
                # Let's try to stick to the original TableLoRA logic logic but be careful.
                # If you encounter "dimension mismatch" errors, check this slicing.
                
                # Header
                ids = tokenizer.encode(table_text.split("<ROW>")[0])
                # Safety check: does it start with BOS?
                if len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
                     ids = ids[1:]
                table_token_ids.extend(ids)
                
                row_ids.extend([0] * len(ids))
                col_ids.extend([0] * len(ids))
                
                row_id = 1
                for row_text in table_text.split("<ROW>")[1:]:
                    col_id = 0
                    row_part = "<ROW>" + row_text.split("<COL>")[0]
                    ids = tokenizer.encode(row_part)
                    if len(ids) > 0 and ids[0] == tokenizer.bos_token_id: ids = ids[1:]
                    
                    table_token_ids.extend(ids)
                    row_ids.extend([row_id] * len(ids))
                    col_ids.extend([0] * len(ids))
                    col_id += 1
                    
                    for col_text in row_text.split("<COL>")[1:]:
                        col_part = "<COL>" + col_text
                        ids = tokenizer.encode(col_part)
                        if len(ids) > 0 and ids[0] == tokenizer.bos_token_id: ids = ids[1:]
                        
                        table_token_ids.extend(ids)
                        row_ids.extend([row_id] * len(ids))
                        col_ids.extend([col_id] * len(ids))
                        col_id += 1
                    row_id += 1

                tab_id = tokenizer.convert_tokens_to_ids("<TAB>")
                if input_ids.count(tab_id) == 0:
                    break 
                table_insert_idx = input_ids.index(tab_id)
                
                row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
                col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
                input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]
                
            model_inputs["row_ids"].append(row_ids_total)
            model_inputs["col_ids"].append(col_ids_total)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        else: # Markdown format
            row_ids_total = [0] * len(input_ids)
            col_ids_total = [0] * len(input_ids)
            for table_text in table_texts:
                table_token_ids = []
                col_ids = []
                row_ids = []
                row_id = 1
                
                for row_text in table_text.split("\n"):
                    if set(row_text.strip()) <= set(["|", "-", ":", " "]):
                        row_id -= 1
                        ids = tokenizer.encode(row_text + "\n")
                        if len(ids) > 0 and ids[0] == tokenizer.bos_token_id: ids = ids[1:]
                        
                        table_token_ids.extend(ids)
                        row_ids.extend([row_id] * len(ids))
                        col_ids.extend([0] * len(ids))
                        row_id += 1
                    else:
                        # Row prefix
                        prefix = "|" + "|".join(row_text.split("|")[:2]) + "|"
                        ids = tokenizer.encode(prefix)
                        if len(ids) > 0 and ids[0] == tokenizer.bos_token_id: ids = ids[1:]
                        
                        table_token_ids.extend(ids)
                        row_ids.extend([row_id] * len(ids))
                        col_ids.extend([0] * len(ids))
                        col_id = 1
                        
                        for col_text in row_text.split("|")[2:]:
                            cell = col_text + "|"
                            ids = tokenizer.encode(cell)
                            if len(ids) > 0 and ids[0] == tokenizer.bos_token_id: ids = ids[1:]
                            
                            table_token_ids.extend(ids)
                            row_ids.extend([row_id] * len(ids))
                            col_ids.extend([col_id] * len(ids))
                            col_id += 1
                        row_id += 1

                tab_id = tokenizer.convert_tokens_to_ids("<TAB>")
                if input_ids.count(tab_id) == 0:
                    break 
                table_insert_idx = input_ids.index(tab_id)
                
                row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
                col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
                input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]

            model_inputs["row_ids"].append(row_ids_total)
            model_inputs["col_ids"].append(col_ids_total)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

    if getattr(data_args, "emb_lora", False):
        tokenizer = ori_tokenizer
        
    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
