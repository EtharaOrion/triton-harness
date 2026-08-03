
import tqdm
import os
import argparse
import pandas as pd
import json

from repocoder import eval_repo_coder
from naive_rag import eval_rag_project
from AIClient import OpenAIClient, BaseAIClient, AnthropicBridgeClient

def eval_rag(rag_type: str, llm: BaseAIClient, language_list=['py', 'java', 'go'], limit: int = 0):

    if any([1 for l in language_list if l not in ['py', 'java', 'go']]):
        raise ValueError('only support py, java, go')
    
    if rag_type not in ['repo_coder', 'embedding', 'bm25', 'mix']:
        raise ValueError('only support repo_coder, embedding, bm25, mix')
    
    with open('./config.json', 'r') as f:
        config = json.load(f)
        
    for lan in language_list:
        repo_root = f'../repo/{lan}_data'
        dataset = pd.read_excel(f'../data/{lan}_data_final.xlsx')
        if limit and limit > 0:
            dataset = dataset.head(limit)
        predict_result = {}
        for project, dataset_df in dataset.groupby('project'):
            if rag_type == 'repo_coder':
                cur_result = eval_repo_coder(dataset_df, project, llm, 
                                            repo_root, config['repo_coder'], lan)
                predict_result.update(cur_result)
            else:
                cur_result = eval_rag_project(rag_type, llm, repo_root, 
                                            dataset_df, lan, config['naive_rag'])
                predict_result.update(cur_result)
        os.makedirs('../result/rag', exist_ok=True)
        with open(f'../result/rag/{rag_type}_{lan}.json', 'w') as f:
            json.dump(predict_result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-rag_type', type=str, help='rag method, should be: repo_coder, embedding, bm25 or mix')
    parser.add_argument('-lang_list', type=str, help='languages, comma separated, one or more in [py, java, go]')
    parser.add_argument('-model_name', type=str, help='model name, any model using open ai client')
    parser.add_argument('-limit', type=int, default=0, help='if >0, only process the first N rows per language (quick test runs)')
    args = parser.parse_args()

    rag_type = args.rag_type
    language_list = args.lang_list.split(',')
    model_name = args.model_name

    with open('./config.json', 'r') as f:
        config = json.load(f)
    config = config['ai_client']
    if config['url'] == "" or config['key'] == "":
        raise NotImplementedError("Please provide the url and key for the AI client in config.json")
    if config.get('provider') == "anthropic_bridge":
        llm = AnthropicBridgeClient(url=config['url'], key=config['key'], model=model_name)
    else:
        llm = OpenAIClient(url=config['url'], key=config['key'], model=model_name)
    eval_rag(rag_type, llm, language_list, limit=args.limit)

