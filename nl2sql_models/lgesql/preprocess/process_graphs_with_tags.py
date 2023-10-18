#coding=utf8
from copy import deepcopy
import os, json, pickle, argparse, sys, time

from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocess.graph_utils import GraphProcessor

NUM = 1
def process_dataset_graph(processor, dataset, tables, method, output_path=None, skip_large=False):
    processed_dataset = []
    for idx, entry in tqdm(enumerate(dataset[::NUM]), total=len(dataset[::NUM])):
        db = tables[entry['db_id']]
        if skip_large and len(db['column_names']) > 100:
            continue
        if (idx + 1) % 500 == 0:
            print('Processing the %d-th example ...' % (idx + 1))
        entry = processor.process_graph_utils(entry, db, method=method)
        for entry1 in dataset[idx*NUM: (idx+1)*NUM]:
            entry1['graph'] = entry['graph']
            processed_dataset.append(entry1)
    print('In total, process %d samples, skip %d samples .' % (len(processed_dataset), len(dataset) - len(processed_dataset)))
    if output_path is not None:
        # serialize preprocessed dataset
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--table_path', type=str, required=True, help='processed table path')
    arg_parser.add_argument('--method', type=str, default='lgesql', choices=['rgatsql', 'lgesql'])
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    args = arg_parser.parse_args()

    processor = GraphProcessor()
    # loading database and dataset
    tables = pickle.load(open(args.table_path, 'rb'))
    dataset = pickle.load(open(args.dataset_path, 'rb'))
    start_time = time.time()
    dataset = process_dataset_graph(processor, dataset, tables, args.method, args.output_path)
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
