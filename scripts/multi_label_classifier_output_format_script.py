import click
import itertools

from allenmodels.dataset_readers.dataset_utils.query_to_toks import is_number

@click.command()
@click.argument("classifier_preds_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("meta_dict_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(exists=False, dir_okay=False))
def main(classifier_preds_file, meta_dict_file, output_file):
    preds = open(classifier_preds_file)
    lines = preds.readlines()

    # Get the dict that meta as the key, and index as the value
    # Used for getting seralization
    meta_index_dict = {}
    meta = open(meta_dict_file, 'r')
    for i, line in enumerate(meta.readlines()):
        key = line.strip()
        # sort the key
        parts = key.split(', ')
        rating = parts[0]
        tags = sorted(parts[1:])
        new_key = ', '.join([rating] + tags)
        meta_index_dict[new_key] = i

    outputs = []
    # Get the ordered labels both for tags and ratings
    labels = lines[0].strip().split(', ')
    for i, line in enumerate(lines[1::2]):
        probs = line.strip().split(', ')
        threshold = 0.0
        rating_prob = -100.0
        rating = -1
        tags = []
        for p, l in zip(probs, labels):
            if is_number(l):
                if float(p) > rating_prob:
                    rating = int(l)
                    rating_prob = float(p)
                else: continue
            elif float(p) > threshold:
                tags.append(l)
        # If no tag is predicted above zero probability, 
        # lower the threshold until any tag is found
        if not tags:
            while (not tags):
                threshold -= 5
                for p, l in zip(probs, labels):
                    if is_number(l): continue
                    if float(p) > threshold:
                        tags.append(l)
        # Find the combinations that are the same as those in training data               
        tags = sorted(tags)
        ratings = [j for j in range(rating-200, rating+201)[::50]]
        combines = []
        for L in range(1, len(tags)+1):
            for subset in itertools.combinations(tags, L):
                combines.append(sorted(list(subset)))
        res = []
        for r in ratings:
            for c in combines:
                key = str(r) + ', ' + ', '.join(c)
                if key in meta_index_dict.keys(): 
                    res.append(str(meta_index_dict[key]))
                elif c == ['subquery', 'where']:
                    c = ['subquery', 'subquery', 'where']
                    key = str(r) + ', ' + ', '.join(c)
                    if key in meta_index_dict.keys(): 
                        res.append(str(meta_index_dict[key]))
                elif c == ['join', 'subquery', 'where']: 
                    c = ['join', 'subquery', 'subquery', 'where']
                    key = str(r) + ', ' + ', '.join(c)
                    if key in meta_index_dict.keys(): 
                        res.append(str(meta_index_dict[key]))
                        
        outputs.append(', '.join(res))

    with open(output_file, 'w') as f:
        for o in outputs:
            f.write(f'{o}\n')

if __name__ == "__main__":
    main()