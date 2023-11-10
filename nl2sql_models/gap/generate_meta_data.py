import click, json, copy

@click.command()
@click.argument("test_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(exists=False, dir_okay=False))
def main(test_file, output_file):
    f = open("./output/metadata.txt", 'r')
    metadata = [line.strip().split(', ') for line in f.readlines()]
    f = open("./meta_dict.txt", 'r')
    meta_dict = {}
    for iidx, line in enumerate(f.readlines()):
        meta_dict[iidx] = line.strip()
    # print(metadata[0])
    with open(test_file) as f:
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

    with open(output_file, 'w') as json_file:
        json.dump(input, json_file,indent=4)

if __name__ == "__main__":
    main()