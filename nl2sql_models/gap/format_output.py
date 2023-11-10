import click, json


@click.command()
@click.argument("output_file", type=click.Path(exists=False, dir_okay=False))
def main(output_file):
    input_file = "nl2sql_models/gap/ie_dirs/bart_run_1_true_1-step41000.infer"
    with open(input_file) as json_file:
        json_list = list(json_file)

    outputs = []
    for json_str in json_list:
        result = json.loads(json_str)
        outputs.append(result['beams'][0]['inferred_code'])

    with open(output_file, "w") as f:
        for o in outputs:
            f.write(f"{o}\n")

if __name__ == "__main__":
    main()