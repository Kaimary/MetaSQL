import re
from typing import Dict, List, Set
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy as sp
from spacy.tokens import Doc
from spacy.symbols import ORTH

STOP_WORDS = set(stopwords.words('english'))
SQL_KEYWORDS = [
    'count', 'sum', 'avg', 'max', 'min', 'group', 'by', 'select', 'from', 'order', 'limit', 'intersect', 'union',
    'except', 'join', 'where'
]

nlp = sp.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
special_orths = [
    [{ORTH: "AVG("}],
    [{ORTH: "COUNT("}],
    [{ORTH: "MAX("}],
    [{ORTH: "MIN("}],
    [{ORTH: "SUM("}],
]
special_cases = [
    "AVG(", "COUNT(", "MAX(", "MIN(", "SUM("
]
for case, orth in zip(special_cases, special_orths):
    nlp.tokenizer.add_special_case(case, orth)
porter = PorterStemmer()


def clean_text(text):
    # encoding
    try:
        t = text.encode("ISO 8859-1")
        enc_text = t.decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        enc_text = text

    # line break
    text = enc_text.replace('\n', ' ')

    # empty characters
    text = " ".join(text.strip().split())

    return text


def stemmer(q, is_sql=False):
    # q = nlp(clean_text(query))
    # return [porter.stem(token.text.replace('Ġ', '').lower()) if token.text.lower() not in STOP_WORDS else '' for token in q[1:-1]]
    return ['' if not is_sql and token.text.lower() in STOP_WORDS else WordNetLemmatizer().lemmatize(
        token.text.replace('Ġ', '').lower()) for token in q]  # [1:-1]]


def sql_format(sql: str):
    
    table_alias_pattern = re.compile(r'[a-zA-Z_]+[\w]*\.')
    sql = re.sub(table_alias_pattern, '', sql)

    # *** Place \( in a capture group (...) and then use \2 to refer to it ***
    agg_space_pattern = re.compile(r'([\w]*)(\()')
    sql = re.sub(agg_space_pattern, r'\1\2 ', sql)
    agg_space_pattern1 = re.compile(r'([\w]*)(\))')
    sql = re.sub(agg_space_pattern1, r'\1 \2', sql)
    
    toks = sql.strip().split()
    toks = [t.upper() if t in SQL_KEYWORDS else t for t in toks]

    # max_pattern = re.compile(r'MAX\(([\w|\*]*)\)')
    # sql = re.sub(max_pattern, r'maximum \1', sql)
    # min_pattern = re.compile(r'MIN\(([\w|\*]*)\)')
    # sql = re.sub(min_pattern, r'minimum \1', sql)
    # avg_pattern = re.compile(r'AVG\(([\w|\*]*)\)')
    # sql = re.sub(avg_pattern, r'average \1', sql)
    # sum_pattern = re.compile(r'SUM\(([\w|\*]*)\)')
    # sql = re.sub(sum_pattern, r'summary \1', sql)

    # toks = [t if idx > 0 and toks[idx-1] in ['FROM', 'JOIN'] else t.replace('_', ' ') for idx, t in enumerate(toks)]

    return ' '.join(toks)


def get_query_token_types(stems: List[str], table_set: Set[str], column_set: Set[str]):
    """
    Params:
    stems: A list of stems of a string (including some empty stems which indicate corresponding stop words in original string)
    table_set:   The set of stems of table items in the underlying database schema
    column_set: The set of stems of column items in the underlying database schema

    return: List of type strings, matched stem-table index dictionary and matched stem-column index dictionary
    """
    types = []
    t_i = 0
    c_i = 0
    stem_to_tid = dict()
    stem_to_cid = dict()
    for idx, stem in enumerate(stems):
        if not stem:
            types.append('none')
            continue
        # Match with any table items
        if len(set([stem]) & table_set) != 0:
            # Check if the stem has visited
            if stem in stem_to_tid:
                i = stem_to_tid[stem]
                types.append(f't{i}')
            else:
                stem_to_tid[stem] = t_i
                types.append(f't{t_i}')
                t_i += 1
        elif len(set([stem]) & column_set) != 0:
            # Check if the stem has visited
            if stem in stem_to_cid:
                i = stem_to_cid[stem]
                types.append(f'c{i}')
            # We assume some column includes multiple tokens
            # We mark consecutive columns to be the same one
            elif idx > 0 and stems[idx - 1] in stem_to_cid:
                i = stem_to_cid[stems[idx - 1]]
                # The stem shares the same column index with the previous one
                stem_to_cid[stem] = i
                types.append(f'c{i}')
            else:
                stem_to_cid[stem] = c_i
                types.append(f'c{c_i}')
                c_i += 1
        else:
            types.append('none')

    return ' '.join(types), stem_to_tid, stem_to_cid


def get_dialect_token_types(stems: List[str], stem_to_tid: Dict[str, int], stem_to_cid: Dict[str, int],
                            column_set: Set[str]):
    """
    For the dialect types, we use the table/column indices found in the query string to match with the ones in dialect.
    For those unmatched column tokens in dialect, we use "c[num]*" to distinguish.

    Params:
    stems: A list of stems of a string (including some empty stems which indicate corresponding stop words in original string)
    stem_to_tid: Matched stem-table index dictionary found in the query string
    stem_to_cid: Matched stem-column index dictionary found in the query string
    column_set: The set of stems of column items in the underlying database schema

    return: List of type strings
    """

    types = []
    c_star_i = 0
    stem_to_cstarid = dict()
    cls_type = "select"
    for idx, stem in enumerate(stems):
        if stem in SQL_KEYWORDS: cls_type = stem

        if cls_type in ['from', 'join'] and stem in stem_to_tid.keys():
            t_i = stem_to_tid[stem]
            types.append(f"t{t_i}")
        elif cls_type not in ['from', 'join', 'limit'] and stem in stem_to_cid.keys():
            c_i = stem_to_cid[stem]
            types.append(f"c{c_i}")
            # Check the previous token
            # If the token is recoginized as `c[num]*`, we change it to be the same `c[num]`
            if idx > 0 and stems[idx - 1] in stem_to_cstarid:
                types[idx - 1] = f"c{c_i}"
                # del stem_to_cstarid[stems[idx-1]]
        # Make sure checking the columns but not the tables in sql
        elif cls_type not in ['from', 'join', 'limit'] and len(set([stem]) & column_set) != 0:
            # Check if the stem has visited
            if stem in stem_to_cstarid:
                i = stem_to_cstarid[stem]
                types.append(f'c{i}*')
            # We assume some column includes multiple tokens
            # We mark consecutive columns to be the same one
            elif idx > 0 and stems[idx - 1] in stem_to_cstarid:
                i = stem_to_cstarid[stems[idx - 1]]
                types.append(f'c{i}*')
            elif idx > 0 and stems[idx - 1] in stem_to_cid:
                i = stem_to_cid[stems[idx - 1]]
                types.append(f'c{i}')
            else:
                stem_to_cstarid[stem] = c_star_i
                types.append(f'c{c_star_i}*')
                c_star_i += 1
        else:
            types.append(f"none")

    return ' '.join(types)


# Deprecated
def marker(query, dialects):
    q = nlp(clean_text(query))

    marked_q = []
    stem_to_id = dict()
    q_i = 0
    for token in q:
        marked_q.append(token.text)
        if not (token.is_punct or token.is_stop):
            stem = porter.stem(token.text.lower())
            marked_q.pop()
            if stem in stem_to_id:
                i = stem_to_id[stem]
                marked_q.append(f"[e{i}]{token.text}[\e{i}]")
            if stem not in stem_to_id:
                stem_to_id[stem] = q_i
                marked_q.append(f"[e{q_i}]{token.text}[\e{q_i}]")
                q_i += 1

    doc = []
    for dialect in dialects:
        d = nlp(clean_text(dialect))
        marked_d = []
        for term in d:
            marked_d.append(term.text)
            if not (term.is_punct or term.is_stop):
                stem = porter.stem(term.text.lower())
                for q_stem in stem_to_id:
                    if q_stem == stem:
                        q_i = stem_to_id[stem]
                        marked_d.pop()
                        marked_d.append(f"[e{q_i}]{term.text}[\e{q_i}]")
                        break

        dial = Doc(nlp.vocab, words=marked_d, spaces=[token.whitespace_ for token in d])
        doc.append(''.join(token.text_with_ws for token in dial))

    qu = Doc(nlp.vocab, words=marked_q, spaces=[token.whitespace_ for token in q])
    return ''.join(token.text_with_ws for token in qu), doc
