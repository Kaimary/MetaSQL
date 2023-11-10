#coding=utf8
import sys, os, json, pickle, argparse, time, torch
from argparse import Namespace

from tqdm import tqdm
# These three lines are used in the docker environment rhythmcao/text2sql:v2.0
# os.environ['NLTK_DATA'] = os.path.join(os.path.sep, 'root', 'nltk_data')
# os.environ["STANZA_RESOURCES_DIR"] = os.path.join(os.path.sep, 'root', 'stanza_resources')
# os.environ['EMBEDDINGS_ROOT'] = os.path.join(os.path.sep, 'root', '.embeddings')
from nl2sql_models.lgesql.preprocess.process_dataset_with_tags import process_tables, process_dataset
from nl2sql_models.lgesql.preprocess.process_graphs_with_tags import process_dataset_graph
from nl2sql_models.lgesql.preprocess.common_utils import Preprocessor
from nl2sql_models.lgesql.preprocess.graph_utils import GraphProcessor
from nl2sql_models.lgesql.utils.example import Example
from nl2sql_models.lgesql.utils.batch import Batch
from nl2sql_models.lgesql.model.model_utils import Registrable
from nl2sql_models.lgesql.model.model_constructor import *

def preprocess_dataset(processor, dataset, metadata, meta_dict, tables, method='lgesql'):
    output_dataset = process_dataset(processor, dataset, metadata, meta_dict, tables)
    graph_processor = GraphProcessor()
    output_dataset = process_dataset_graph(graph_processor, output_dataset, tables, method=method)
    return output_dataset

def preprocess_database(processor, table_path='data/tables.json', saved_table_path=''):
    if not saved_table_path:
        tables = json.load(open(table_path, 'r'))
        output_tables = process_tables(processor, tables)
    else:
        output_tables = pickle.load(open(saved_table_path, 'rb'))
    return output_tables

def load_examples(dataset, tables):
    ex_list = []
    for ex in dataset:
        ex_list.append(Example(ex, tables[ex['db_id']]))
    return ex_list

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_dict_path', default='metadata.txt', help='path to metadata dict file')
parser.add_argument('--metadata_output_path', default='outputs.txt', help='path to metadata output file')
parser.add_argument('--db_dir', default='database', help='path to db dir')
parser.add_argument('--table_path', default='data/tables.json', help='path to tables json file')
parser.add_argument('--saved_table_path', default='data/tables.bin', help='path to tables bin file')
parser.add_argument('--dataset_path', default='data/dev.json', help='path to raw dataset json file')
parser.add_argument('--saved_model', default='saved_models/glove42B', help='path to saved model path, at least contain param.json and model.bin')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=5, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
args = parser.parse_args(sys.argv[1:])

params = json.load(open(os.path.join(args.saved_model, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PLM from AutoConfig instead of AutoModel.from_pretrained(...)
# Preprocess database
print('Preprocess database ...')
processor = Preprocessor(db_dir=args.db_dir, db_content=True)
preprocessed_tables = preprocess_database(processor, table_path=args.table_path, saved_table_path=args.saved_table_path)
Example.configuration(plm=params.plm, method=params.model, tables=preprocessed_tables, table_path=args.table_path, db_dir=args.db_dir)
evaluator = Example.evaluator
# Load model checkpoint
print('Load model checkpoint ...')
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
model = Registrable.by_name('text2sql')(params, Example.trans).to(device)
check_point = torch.load(open(os.path.join(args.saved_model, 'model.bin'), 'rb'), map_location=device)
model.load_state_dict(check_point['model'])
print('Load full dataset ...')
dataset = json.load(open(args.dataset_path, 'r'))
f = open(args.metadata_output_path, 'r')
metadata = [line.strip().split(', ') for line in f.readlines()]
f = open(args.metadata_dict_path, 'r')
meta_dict = {}
for iidx, line in enumerate(f.readlines()):
    meta_dict[iidx] = line.strip()
# Preprocess dataset by each database
bstart = False
# cur_db_id = 'school_player' #dataset[0]['db_id']
cur_db_id = dataset[0]['db_id']
one_dataset = []
one_dataset_metadata = []
assert len(dataset) == len(metadata)
for ex, meta in zip(dataset, metadata):
    if not bstart and ex['db_id'] != dataset[0]['db_id']: continue
    bstart = True
    if ex['db_id'] == cur_db_id: 
        one_dataset.append(ex)
        one_dataset_metadata.append(meta)
        continue
    print(f'Preprocess dataset `{cur_db_id}` ...')
    # Preprocessed one dataset
    preprocessed_dataset = preprocess_dataset(processor, one_dataset, one_dataset_metadata, meta_dict, preprocessed_tables, method=params.model)
    preprocessed_dataset = load_examples(preprocessed_dataset, preprocessed_tables)
    start_time = time.time()
    print('Start evaluating ...')
    model.eval()
    all_hyps = []
    with torch.no_grad():
        for i in tqdm(range(0, len(preprocessed_dataset), args.batch_size)):
            current_batch = Batch.from_example_list(preprocessed_dataset[i: i + args.batch_size], device, train=False)
            hyps = model.parse(current_batch, args.beam_size)
            all_hyps.extend(hyps)
    with open(args.output_path, 'a', encoding='utf8') as of:
        for idx, hyp in tqdm(enumerate(all_hyps), total=len(all_hyps)):
            pred_sql = evaluator.obtain_sql(hyp, preprocessed_dataset[idx].db)
            # best_ast = hyp[0].tree # by default, the top beam prediction
            # pred_sql = Example.trans.ast_to_surface_code(best_ast, preprocessed_dataset[idx].db)
            of.write(pred_sql + '\n')
    print('Evaluation costs %.4fs .' % (time.time() - start_time))
    #if ex['db_id'] == 'perpetrator': break
    # Switch to next dataset
    cur_db_id = ex['db_id']
    one_dataset.clear()
    one_dataset_metadata.clear()
    one_dataset.append(ex)
    one_dataset_metadata.append(meta)

# Processing the instances of the final database
print(f'Preprocess dataset `{cur_db_id}` ...')
# Preprocessed one dataset
preprocessed_dataset = preprocess_dataset(processor, one_dataset, one_dataset_metadata, meta_dict, preprocessed_tables, method=params.model)
preprocessed_dataset = load_examples(preprocessed_dataset, preprocessed_tables)
start_time = time.time()
print('Start evaluating ...')
model.eval()
all_hyps = []
with torch.no_grad():
    for i in tqdm(range(0, len(preprocessed_dataset), args.batch_size)):
        current_batch = Batch.from_example_list(preprocessed_dataset[i: i + args.batch_size], device, train=False)
        hyps = model.parse(current_batch, args.beam_size)
        all_hyps.extend(hyps)
with open(args.output_path, 'a', encoding='utf8') as of:
    for idx, hyp in tqdm(enumerate(all_hyps), total=len(all_hyps)):
        pred_sql = evaluator.obtain_sql(hyp, preprocessed_dataset[idx].db)
        # best_ast = hyp[0].tree # by default, the top beam prediction
        # pred_sql = Example.trans.ast_to_surface_code(best_ast, preprocessed_dataset[idx].db)
        of.write(pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
