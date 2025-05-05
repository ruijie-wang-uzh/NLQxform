from argparse import ArgumentParser
import requests
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import time
import re
from SPARQLWrapper import SPARQLWrapper, JSON

'''
do query and generate answer file for submission
'''


def get_answer(x: dict):
    l = []
    if "results" in x.keys():
        ll = list(x["results"]["bindings"])
        for one in ll:
            if 'answer' in one.keys():
                l.append(one["answer"]["value"])
            elif "count" in one.keys():
                l.append(one["count"]["value"])
            elif "firstanswer" in one.keys() and "secondanswer" in one.keys():
                l.append(one["firstanswer"]["value"])
                l.append(one["secondanswer"]["value"])
            else:
                if len(one) == 0:
                    pass
                else:
                    print(one, " IN ", ll)
        return l
    else:
        if "boolean" in x.keys():
            l.append(x["boolean"])  # True or False
        return l


def do_query(query):
    sparql.setQuery(f"""
    {query}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


def pipeline(path):
    df = pd.read_json(path)
    l = []
    ll = []
    if "potential_query" in df.keys():
        for i, row in tqdm(df.iterrows()):
            result = None
            simplified = []
            queries = row["potential_query"]
            if queries and len(queries) > 0:
                for query in queries:
                    result = do_query(query)
                    if "results" in result.keys():
                        if len(result["results"]["bindings"]) == 0:
                            pass
                        else:
                            simplified = get_answer(result)
                            if len(simplified) > 0:
                                break
                    elif "boolean" in result.keys():
                        simplified = get_answer(result)
                        if len(simplified) > 0:
                            break
            l.append(result)
            ll.append(simplified)
    df["result"] = l
    df["answer"] = ll
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resume_dir", default='logs', type=str)
    parser.add_argument("--in_name", type=str, help="save path for output dataframe saved as json file",
                        default="postprocess_heldout.json")
    parser.add_argument("--default_dir", type=str, default="./",
                        help="default_dir", dest="default_dir")
    parser.add_argument("--resume_prefix", type=str)
    parser = parser.parse_args()

    sparql_endpoint = "https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql"
    sparql = SPARQLWrapper(sparql_endpoint)

    df = pipeline(os.path.join(parser.default_dir, parser.resume_dir, parser.resume_prefix, parser.in_name))

    lll = []
    for i, row in df.iterrows():
        l = []
        for k, v in row["mappings"].items():
            l.extend(v)
        l = ["<" + a + ">" for a in l]
        lll.append(list(set(l)))
    df["entities"] = lll

    sub = df[["id", "answer", "entities"]]

    result_dict = [value for index, value in enumerate(sub.to_dict(orient='records'))]

    with open(os.path.join(parser.default_dir, parser.resume_dir, parser.resume_prefix, "answer.txt"),
              'w') as json_file:
        json.dump(result_dict, json_file, indent=6)
