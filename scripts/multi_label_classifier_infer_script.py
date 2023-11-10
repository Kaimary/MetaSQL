#coding=utf8
import sys, os, json, argparse, time, torch
from argparse import Namespace
from nl2sql_models.lgesql.preprocess.process_dataset import process_tables, process_dataset
from nl2sql_models.lgesql.preprocess.process_graphs import process_dataset_graph
from nl2sql_models.lgesql.preprocess.common_utils import Preprocessor
from nl2sql_models.lgesql.preprocess.graph_utils import GraphProcessor
from nl2sql_models.lgesql.utils.example import Example
from nl2sql_models.lgesql.utils.batch import Batch
from nl2sql_models.lgesql.model.model_utils import Registrable
from nl2sql_models.lgesql.model.model_constructor import *

def preprocess_database_and_dataset(db_dir='database/', table_path='data/tables.json', dataset_path='data/dev.json', method='lgesql'):
    tables = json.load(open(table_path, 'r'))
    dataset = json.load(open(dataset_path, 'r'))
    processor = Preprocessor(db_dir=db_dir, db_content=True)
    output_tables = process_tables(processor, tables, output_path='output/tables.bin')
    output_dataset = process_dataset(processor, dataset, output_tables)
    graph_processor = GraphProcessor()
    output_dataset = process_dataset_graph(graph_processor, output_dataset, output_tables, method=method)
    return output_dataset, output_tables

def load_examples(dataset, tables):
    ex_list = []
    for ex in dataset:
        ex_list.append(Example(ex, tables[ex['db_id']]))
    return ex_list

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='database', help='path to db dir')
parser.add_argument('--table_path', default='data/tables.json', help='path to tables json file')
parser.add_argument('--dataset_path', default='data/dev.json', help='path to raw dataset json file')
parser.add_argument('--saved_model', default='saved_models/glove42B', help='path to saved model path, at least contain param.json and model.bin')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=1, type=int, help='batch size for evaluation')
parser.add_argument('--use_gpu', action='store_true', help='whether use gpu')
args = parser.parse_args(sys.argv[1:])

params = json.load(open(os.path.join(args.saved_model, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PLM from AutoConfig instead of AutoModel.from_pretrained(...)
dataset, tables = preprocess_database_and_dataset(db_dir=args.db_dir, table_path=args.table_path, dataset_path=args.dataset_path, method=params.model)
Example.configuration(plm=params.plm, method=params.model, tables=tables, table_path=args.table_path, db_dir=args.db_dir)
dataset = load_examples(dataset, tables)

device = torch.device(0) if torch.cuda.is_available() and args.use_gpu else torch.device("cpu")
model = Registrable.by_name('text2label')(params).to(device)
check_point = torch.load(open(os.path.join(args.saved_model, 'model.bin'), 'rb'), map_location=device)
model.load_state_dict(check_point['model'])

start_time = time.time()
print('Start evaluating ...')

label_dict = open("label_dict.txt")
labels = [line.strip() for line in label_dict.readlines()]
model.eval()
res = []
with torch.no_grad():
    for i in range(0, len(dataset), args.batch_size):
        current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
        logits = model.test(current_batch)
        probs = []
        for p in logits[0]:
            probs.append(f"{p:.2f}")
        res.append(probs)
with open(args.output_path, 'w', encoding='utf8') as of:
    for l in res:
        of.write(', '.join(labels) + '\n')
        of.write(', '.join(l) + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
