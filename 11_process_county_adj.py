import pandas as pd

FN_CADJ = "data/raw/county_adj.txt"

county_adj_map = {}
with open(FN_CADJ, "r") as f:
    from_county = None
    for line in f.readlines():
        line = line.replace('\n', "").split("\t")
        if line[0] != '':
            # we have a new 'from county'
            from_county = line[1]
            # theres always a first adj on the same line?
            county_adj_map[from_county] = [line[3]]
        else:
            # we're adding to a 'current' from_county
            county_adj_map[from_county].append(line[3])

print(county_adj_map['39165'])
