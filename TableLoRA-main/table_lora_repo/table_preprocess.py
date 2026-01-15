# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

# 移除顶部的全局导入，改为在 main 中按需导入，防止其他数据集缺失文件导致报错
# from table_preprocess.wikitq import preprocess_wikitq
# from table_preprocess.fetaqa import preprocess_fetaqa
# from table_preprocess.tabfact import preprocess_tabfact
# from table_preprocess.hitab import preprocess_hitab

# paser
def parse_args():
    parser = argparse.ArgumentParser(description='Extract code snippets from a given codebase')
    parser.add_argument('--dataset_name', type=str, default='wikitq', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='Model name or path')
    parser.add_argument('--max_length', type=int, default=1024, help='Max length of the prompt')
    parser.add_argument('--prompt_tuning', type=bool, default=False, help='Use prompt tuning or not')
    parser.add_argument('--save_original_table', type=bool, default=False, help='Save original table or not')
    parser.add_argument('--pretraining', type=bool, default=False, help='Pretraining or not')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.dataset_name == 'wikitq':
        # 仅当需要跑 wikitq 时才导入，安全避开其他数据集的报错
        from table_preprocess.wikitq import preprocess_wikitq
        preprocess_wikitq(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table, args.pretraining)
        
    elif args.dataset_name == 'fetaqa':
        from table_preprocess.fetaqa import preprocess_fetaqa
        preprocess_fetaqa(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table)
        
    elif args.dataset_name == 'tabfact':
        from table_preprocess.tabfact import preprocess_tabfact
        preprocess_tabfact(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table, args.pretraining)
        
    elif args.dataset_name == 'hitab':
        # 如果缺少 hitab 相关文件，只要不跑这个分支就不会报错
        from table_preprocess.hitab import preprocess_hitab
        preprocess_hitab(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table, args.pretraining)
        
    else:
        print('Dataset not supported')