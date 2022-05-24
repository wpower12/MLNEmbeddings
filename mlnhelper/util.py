

def str_to_tuplelist(s):
    if s == "[]":
        return []
    tuples = s.replace("[", "").replace("]", "").replace("), (", "):(").split(":")
    tuples = [s.replace("(", "").replace(")", "") for s in tuples]
    tuples = [s.split(", ") for s in tuples]
    tuples = [(int(s[0]), int(s[1])) for s in tuples]
    return tuples
