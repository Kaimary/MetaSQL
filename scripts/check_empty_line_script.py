import click

@click.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def main(input_file):
    input = open(input_file)
    lines = input.readlines()
    empty_line = False
    prev = lines[0].strip()
    new_lines = [prev]
    for line in lines[1:]:
        if not line.strip():
            empty_line = True
            new_lines.append(prev)
        else:
            new_lines.append(line.strip())
            prev = line.strip()
    
    # Rewrite
    if empty_line:
        with open(input_file, "w") as f:
            for l in new_lines:
                f.write(f"{l}\n")

if __name__ == "__main__":
    main()