import json
import copy

def get_meta_inputfile():
    f = open("./output/metadata.txt", 'r')
    metadata = [line.strip().split(', ') for line in f.readlines()]
    f = open("./meta_dict.txt", 'r')
    meta_dict = {}
    for iidx, line in enumerate(f.readlines()):
        meta_dict[iidx] = line.strip()
    print(metadata[0])
    with open("./data/dev.json") as f:
       ff=json.load(f)
    input=[]
    for index,metas in enumerate(metadata):       
        for meta in metas:  
            temp=copy.deepcopy(ff[index])          
            d=meta_dict[int(meta)].split(",",1)
            temp['rating']=d[0]
            temp['tags']=d[1]
            temp['flag']="CORRECT SOLUTION"
            input.append(temp)

    with open("./data/dev_for_resd.json", 'w') as json_file:
        json.dump(input, json_file,indent=4)

    

get_meta_inputfile()