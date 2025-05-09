import openai
import sys
import json
import pandas as pd

openai.api_key = "INPUT YOUR OPENAI KEY"

component1=["where", "group", "order", "limit", "join", "or", "like"]
component2=["intersect","union","except"]

if sys.argv[1] == "--dataset" and sys.argv[3] == "--output":
    DATASET_SCHEMA = sys.argv[2]+"tables.json"
    DATASET = "spider_dev_with_meta.json"
    OUTPUT_FILE = sys.argv[4]
    DATASET_TRAIN=sys.argv[2]+"train_spider.json"
else:
    raise Exception("Please use this format python openai_generate_with_meta.py --dataset spider/ --output predicted_sql.txt")

def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
  return output

def find_primary_keys_MYSQL_like(db_name):
  df = spider_primary[spider_primary['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output

def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ' with columns '
    for index, row in group.iterrows():
      output += row[" Field Name"]+', '
    output = output[:-1]
    output += "\n"
  return output

def read_meta():
  path="meta_prompt.txt"
  with open(path, 'r') as file:
    # 读取文件内容为字符串
    file_content = file.read()
  return file_content

def process_tags(tags):
  
  tags=tags.replace("\n"," ").split(',')
  t=[]
  for tag in tags:
      tag=tag.replace(" ","")
      if tag=="compound":
          tag="use INTERSECT/EXCEPT/UNION to connect two clause"
      if tag=="order":
          tag="Sorting(ORDER BY) of results is required"
      if tag=="group":
          tag="Group by calculation is needed"
      if tag=="subquery":
          tag= "Solve the sub problems first(subquery) "
      if tag=="join":
          tag="Use join to connect the tables(join)"
      if tag=="where":
          tag="filtering records using some restrictions(where)"
      if tag=="none":
          tag="very easy query,think straight forward"
          
      t.append(tag)
  t=" and ".join(t)
  # print(t)
  return t   

def generate(test_sample_text,database,sql,val_df,tags,rating):
  instruction = "#### Text2SQL task: Give you database schema, NL question and metadata information of the target SQL, generate an executable SQL query for me."
  # In-context learning result
  tags=tags.replace("\n"," ")
  tags=process_tags(tags)
  five_shots=read_meta()
  five_shots+="\n####Please follow the previous example and help me generate the following SQL statement "+ '\n' 
  fields = find_fields_MYSQL_like(database)
  with open('schema.json', 'r') as file:
    data = json.load(file)
  if database in data:
    data=data[database]
    for key in data:
      if key!="sql":
          fields+=f"table {key}: {data[key]}"+"\n"
  advise= "\nfollowing are some advice:\n1 only use group by on one column only;\n 2 only include count(*) if clearly required in questions,\n don't rename column"
  prompt = instruction+five_shots+ fields+ '#### Question: ' + test_sample_text 
  prompt+=f'\n#### The sql must satisfied:  {tags}, The difficulty rating of the target SQL is {rating}'
  prompt+='\n#### SQLite SQL QUERY\nSELECT'
  return prompt

def GPT4_debug(prompt):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
       {"role": "system", "content": "you are a database expert"},
       {"role": "user", "content": prompt}],
    n = 1,
    stream = False,
    temperature=0.0,
    max_tokens=350,
    top_p = 1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop = ["\n\n"]
  )
  return response['choices'][0]['message']['content']

def load_data(DATASET):
    return pd.read_json(DATASET)

def group_cot(sql):
  sql=sql.split()
  # print(sql[2].split("."))
  s=''
  sql=sql[2].split(".")
  if len(sql)==2:
    s=f"group by column {sql[1]} from table {sql[0]}"+'\n' 
  elif len(sql)==1:
     s=f"group by column {sql[0]}"+'\n' 
  return s

def order_cot(sql):
  sql=sql.split()[2]
  sql=sql.split(".")
  s=''
  if len(sql)==2:
    s=f"rank by column {sql[1]} from table {sql[0]}"+'\n' 
  elif len(sql)==1:
     s=f"rank by column {sql[0]}"+'\n' 
  return s

def shots(test_sample_text,database,sql,gold,flag):
  # 2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
    fields=""
    if flag:
      fields = find_fields_MYSQL_like(database)
      fields += "Foreign_keys = " + find_foreign_keys_MYSQL_like(database) + '\n'
      fields += "Primary_keys = " + find_primary_keys_MYSQL_like(database)+ '\n'
    cot='#### Please think step by step.To sovle the problems,we should: \n'
    if 'NESTED_CLAUSE' in sql:
          cot+="Solve the sub problems first,so the NESTED_CLAUSE is: "+ sql['NESTED_CLAUSE']+'\n'
    if 'FROM' in sql:
          table,col=get_tables_and_joinkey(sql['FROM'])
          cot+=f"Choose the table from the table list:{table} ,"+"\n"
          if len(col)!=0:
            cot+="Use join to connect the tables based on the column: "+str(col) +'\n'         
          cot+="so the FROM clause is: "+sql['FROM']+'\n'
    if 'WHERE' in sql:
          cot+="Then consider filtering records using some restrictions, so the WHERE clause is: "+sql['WHERE']+'\n'
    if 'GROUP' in sql:
          s=group_cot(sql['GROUP'])
          cot+=f"Group calculation is needed, select column for grouping,{s}so the GROUPBY clause is:"+sql['GROUP']+'\n'
    if 'HAVING' in sql:
          cot+="Filtering is required for the grouping results, so the HAVING clause is: "+sql['HAVING']+'\n'
    if 'ORDER' in sql:
          s=order_cot(sql['ORDER'])
          cot+=f"Sorting of results is required,{s}so the ORDERBY clause is: "+sql['ORDER']+'\n'
    if 'SELECT' in sql:
          cot+="Select the columns that the questions needed,so the SELECT clause is: "+sql['SELECT']+'\n'
    if 'INTERSECT' in sql:
          cot+="Intersect is needed,the IUE result is: "+str(sql['INTERSECT'])+'\n'
    if 'UNION' in sql:
          cot+="UNION is needed,the IUE result is: "+str(sql['UNION'])+'\n'
    if 'EXCEPT' in sql:
          cot+="Intersect is needed,the IUE result is: "+str(sql['EXCEPT'])+'\n'
     
    prompt = fields+ '#### Question: ' + test_sample_text +'\n'+cot+'#### To sumup,THE SQLite SQL QUERY: ' + gold+'\n'+'\n'
    return prompt    

if __name__=='__main__':
    spider_schema,spider_primary,spider_foreign=creating_schema(DATASET_SCHEMA)
    val_df = load_data(DATASET)
    val_df_train=load_data(DATASET_TRAIN)
    print(f"Number of data samples {val_df.shape[0]}")
    CODEX = []
    i=0
    for index, row in val_df.iterrows():
      debugged_SQL = None
      while debugged_SQL is None:
          try:
              debugged_SQL = GPT4_debug(generate(row['question'], row['db_id'], row['query'],val_df_train,row["tags"],row["rating"])).replace("\n", " ")
          except Exception as e:
              print(e)
              
      SQL = "SELECT " + debugged_SQL
      print(SQL)
      print(i)
      CODEX.append([row['question'], SQL, row['query'], row['db_id']])

      if len(CODEX)%5==1:
          df = pd.DataFrame(CODEX, columns=['NLQ', 'PREDICTED SQL', 'GOLD SQL', 'DATABASE'])
          question=df['NLQ'].tolist()
          results = df['PREDICTED SQL'].tolist()
          gold = df['GOLD SQL'].tolist()
          FinalResult=[]
          for t in range(len(results)):
              tmp=[]
              tmp.append(question[t])
              tmp.append(results[t])
              tmp.append(gold[t])
              FinalResult.append(tmp)
          with open(OUTPUT_FILE, 'a') as f:
              for t in range(len(FinalResult)):
                  f.write(f"{FinalResult[t][0]}\n")
                  f.write(f"{FinalResult[t][1]}\n")
                  f.write(f"{FinalResult[t][2]}\n")
                  f.write(f"\n")
          CODEX=[]
        



   