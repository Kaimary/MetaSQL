# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 10:20
# @Author  : 
# @Email   : 
# @File    : syntactic_evaluate.py
# @Software: PyCharm

SYNTACTIC_LEVELS = ('nested', 'negation', 'order', 'group', 'having', 'others')
NEGATION_KEYWORDS = {'!=', '! =', ' NOT ', ' EXCEPT '}
IUE_KEYWORDS = {'INTERSECT', 'UNION', 'EXCEPT'}
SYNTACTIC_FUNCTIONS = list()

SYNTACTIC_FUNCTIONS.append(lambda sql: 'nested' if sql.count(' SELECT ') > 1 else '')
SYNTACTIC_FUNCTIONS.append(lambda sql: 'negation' if sum([sql.count(key) for key in NEGATION_KEYWORDS]) else '')
SYNTACTIC_FUNCTIONS.append(lambda sql: 'order' if sql.count(' ORDER BY ') else '')
SYNTACTIC_FUNCTIONS.append(lambda sql: 'group' if sql.count(' GROUP BY ') else '')
SYNTACTIC_FUNCTIONS.append(lambda sql: 'having' if sql.count(' HAVING ') else '')


def syntactic_sql_formatted(sql: str):
    """
    fold sql to ignore the information in subquery
    @param sql: original sql
    @return: formatted sql
    """
    sql = sql.replace('(', ' ( ')
    sql = sql.replace(')', ' ) ')
    tokens = sql.upper().split()
    formatted_tokens = list()
    i = 0
    bracket_nums = 0
    while i < len(tokens):
        # skip brackets
        if tokens[i] == '(':
            bracket_nums += 1
        elif tokens[i] == ')':
            bracket_nums -= 1
        # use select stand for sub query
        elif tokens[i] == 'SELECT' or bracket_nums < 1:
            formatted_tokens.append(tokens[i])

        # use iue keywords and select stand for sub query
        if tokens[i] in IUE_KEYWORDS and bracket_nums < 1:
            formatted_tokens.append('SELECT')
            # no need for more information, break out
            break
        i += 1
    # add withe space at the beginning of the sql
    return ' ' + ' '.join(formatted_tokens) + ' '


def syntactic_evaluate(sql: str):
    """
    classify different sql syntactic category
    @param sql: original sql
    @return: category list, which might be more than one category
    """
    # format sql to evaluate
    sql = syntactic_sql_formatted(sql)
    res = list()
    #  iter in every category
    for syntactic_function in SYNTACTIC_FUNCTIONS:
        one_category = syntactic_function(sql)
        if one_category:
            res.append(one_category)
    if not res:
        res.append('others')
    return res


def test():
    test_sql = 'select T1.name, T2.name from student as T1 join teacher as T2 on T1.mentor_id = T2.id' \
               ' where T1.age < (select avg(age) from student) order by T1.gpa desc limit 1 ' \
               'except select T1.name, T2.name from student as T1 join teacher as T2 on T1.mentor_id = T2.id ' \
               'where T1.age < (select avg(age) from student) order by T1.gpa limit 1;'
    assert syntactic_evaluate(test_sql) == ['nested', 'negation', 'order']


def test2():
    test_sql = 'SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode ' \
               'WHERE T2.Language  =  "English" ' \
               'INTERSECT SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode' \
               ' WHERE T2.Language  =  "French"'
    assert syntactic_evaluate(test_sql) == ['nested']
