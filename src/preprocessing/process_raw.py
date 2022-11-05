import os
import json


# raw data preprocessing functions which convert dstc9 dataset into plain text files.
def split_test_domain_final(data_folder, new_test_folder):
    with open(os.path.join(data_folder, "data_eval", "test", "logs.json"), "r") as f:
        entries = json.load(f)
    with open(os.path.join(data_folder, "data_eval", "knowledge.json"), "r") as f:
        kb = json.load(f)
    with open(os.path.join(data_folder, "data_eval", "test", "labels.json"), "r") as f:
        labels = json.load(f)
    db = []
    for item in kb["attraction"]:
        db.append(kb["attraction"][item]["name"].lower().strip())
    indomain_test = []
    for i, item in enumerate(entries):
        indomain = True
        for turn in item:
            for entity in db:
                if entity in turn["text"].lower():
                    indomain = False
                    break
            if not indomain:
                break
        if indomain:
            indomain_test.append(item)
    os.makedirs(new_test_folder)
    with open(os.path.join(new_test_folder, "test.json"), "w") as f:
        json.dump(indomain_test, f, sort_keys=True, indent=4)


def extract_dstc(filepath, outpath):
    with open(filepath, "r") as f:
        entries = json.load(f)
    db = []
    for item in entries:
        for entry in item:
            db.append(entry["text"] + "\n")
    db = set(db)
    with open(outpath, "w") as f:
        f.writelines(db)