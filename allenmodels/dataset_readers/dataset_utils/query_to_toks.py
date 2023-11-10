# -*- coding: utf-8 -*-
# @Time    : 2022/5/11 15:50
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : query_to_toks.py
# @Software: PyCharm
import re
from typing import List

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False


def to_number(s):
    try:
        return float(s)
    except ValueError:
        pass
    try:
        import unicodedata
        return unicodedata.numeric(s)
    except (TypeError, ValueError):
        pass
    return -99999999999


def query_to_toks_with_value(sql: str) -> List[str]:
    sql = sql.replace("(", " ( ").replace(")", " ) ")\
        .replace(";", "").replace("=",  " = ")\
        .replace(",",  " , ").replace(".", " . ")
    sql = re.sub(r' (\d+)\s+[.]\s+(\d+)\b', r' \1.\2 ', sql)
    sql = re.sub(r' t(\d) . ', r' T\1 . ', sql)
    res = sql.split()
    while '' in res:
        res.remove('')
    return res


def query_to_toks_no_value(sql: str) -> List[str]:
    res = []
    sql = sql.replace("(", " ( ").replace(")", " ) ")\
        .replace(";", "").replace(".", " . ")\
        .replace("=", " = ").replace(",",  " , ")
    sql = re.sub(r' (\d+)\s+[.]\s+(\d+)\b', r' \1.\2 ', sql)
    sql = re.sub(r'\'[a-zA-Z\d\s!.,%-:]+\'', r"value", sql)
    sql = re.sub(r'\"[a-zA-Z\d\s!.,%-:]+\"', r"value", sql)
    toks_no_value = sql.split()
    for tok in toks_no_value:
        if is_number(tok) or (tok.startswith("'") and tok.endswith("'")) or (
                tok.startswith('"') and tok.endswith('"')):
            res.append("value")
        else:
            res.append(tok.lower())
    while '' in res:
        res.remove('')
    return res
