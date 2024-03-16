import requests
from bs4 import BeautifulSoup
import json
from time import sleep
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import time
import logging
import pickle
import re
import difflib
from collections import OrderedDict
import html
import warnings
warnings.filterwarnings("ignore")
from itertools import product
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import argparse


def get_url_by_page(label,verbose=True):
    max_retries = 5
    retry_delay = 0.1
    url="https://dblp.org/search/author?q="+label
    for retry in range(max_retries):
        response = requests.get(url,headers={'Connection':'close'},verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            label_element = soup.select_one('ul.result-list > li > a')
            if label_element:
                label = label_element.get('href')
                return [label]
            elif verbose:
                logger.info("ul.result-list > li > a Element not found on the webpage for url {}.".format(url))
                return []
        elif response.status_code == 429:
            # ttt=retry_delay
            # if retry==0:
            #     ttt=retry_delay
            # elif retry ==1:
            #     ttt=1
            # elif retry ==2:
            #     ttt=30
            # else:
            #     ttt=60
            ttt = float(response.headers['Retry-After'])
                
            if verbose:
                logger.info(f"Rate limited. Retrying in {ttt} seconds...for {url}")
            time.sleep(ttt)
        else:
            if verbose:
                logger.info("Failed to retrieve the webpage for url {}. Status code: {}".format(url,str(response.status_code)))
    return []


def get_url(label_ori,verbose=True):

    # sleep(args.sleep)

    label=label_ori.replace("'","&apos;")
    label_ori=label_ori.replace("'","&apos;")
    for tryt in label.split(" "):
        if tryt.strip().startswith("000") and tryt.strip().isdigit():

            label=label.replace(" "+tryt.strip(),"")

    author_url="https://dblp.org/search/author/api?q="+label+"&h=1000&format=json"
    pub_url="https://dblp.dagstuhl.de/search/publ/api?q="+label+"&h=1000&format=json"
    
    max_retries = 5
    retry_delay = 0.1

    for retry in range(max_retries):
        response = requests.get(author_url,headers={'Connection':'close'},verify=False)
        url=author_url
        if response.status_code == 200:
            data = response.json()
            lll={}
            if 'hit' in data['result']['hits'].keys():
                for o in data['result']['hits']['hit']:
                    if o["info"]["author"] not in lll.keys():
                        lll[o["info"]["author"]]=[]
                    
                    lll[o["info"]["author"]].append(o["info"]["url"])

                if int(data['result']['hits']['@total'])<=1000:
                    tmp=difflib.get_close_matches(label_ori,list(lll.keys()),n=3,cutoff=0.1)
                    if len(tmp)>0:

                        return lll[tmp[0]]
                    else:
                        logger.info(f"null...for {url}")
                        response = requests.get(pub_url)
                        url=pub_url
                else:
                    print("invoke get_url_by_page")
                    return get_url_by_page(label)
               
            else:
                response = requests.get(pub_url,headers={'Connection':'close'},verify=False)
                url=pub_url
        if response.status_code == 200:
            data = response.json()
            lll={}
            if 'hit' in data['result']['hits'].keys():
                for o in data['result']['hits']['hit']:
                    if o["info"]["title"] not in lll.keys():
                        lll[o["info"]["title"]]=[]
                    
                    lll[o["info"]["title"]].append(o["info"]["url"])
 
                if int(data['result']['hits']['@total'])<=1000:
                    tmp=difflib.get_close_matches(label_ori,list(lll.keys()),n=3,cutoff=0.1)
                    if len(tmp)>0:

                        return lll[tmp[0]]
                    else:
                        logger.info(f"null...for {url}")
                else:
                    tmp=difflib.get_close_matches(label,list(lll.keys()),n=3,cutoff=0.1)
                    if len(tmp)>0:

                        return lll[tmp[0]]
                    else:
                        logger.info(f"null...for {url}")

        elif response.status_code == 429:
            # ttt=retry_delay
            # if retry==0:
            #     ttt=retry_delay
            # elif retry ==1:
            #     ttt=1
            # elif retry ==2:
            #     ttt=30
            # else:
            #     ttt=60
            ttt = float(response.headers['Retry-After'])
                
            if verbose:
                logger.info(f"Rate limited. Retrying in {ttt} seconds...for {url}")
            time.sleep(ttt)
        else:
            if verbose:
                logger.info("Failed to retrieve the webpage for url {}. Status code: {}".format(url,str(response.status_code)))
    return []

def prepare():
    data=pd.read_json("data/DBLP-QuAD/processed_train_new.json")
    template_s1=set(list(data["processed_query_template_converted"]))
    data=pd.read_json("data/DBLP-QuAD/processed_valid_new.json")
    template_s2=set(list(data["processed_query_template_converted"]))
    sss=template_s1.union(template_s2)
    sss=list(sss)
    
    initial=["<topic1>","<topic2>","<topic3>"]+["<isnot>","<within>","<num>","<dot>","<dayu>","<xiaoyu>","<comma_sep>","<is_int>","<comma>"]+["<primaryAffiliation>","<yearOfPublication>","<authoredBy>","<numberOfCreators>","<title>","<webpage>","<publishedIn>","<wikidata>","<orcid>","<bibtexType>","<Inproceedings>","<Article>"]
    extra=['?secondanswer', 'GROUP_CONCAT', '?firstanswer', 'separator', 'DISTINCT', '?answer', '?count', 'EXISTS', 'FILTER', 'SELECT', 'STRING1','STRING2', 'BIND','IF', 'COUNT', 'GROUP', 'LIMIT', 'ORDER', 'UNION', 'WHERE', 'DESC','ASC', 'AVG', 'ASK', 'NOT','MAX','MIN','AS', '?x', '?y', '?z', 'BY',"{","}","(",")"]
    
    vocab=initial+extra
    vocab_dict={}
    for i,text in enumerate(vocab):
        vocab_dict[text]='<eid_'+str(i)+'>'
    sss_s=[s.replace(vocab_dict["<num>"],"").replace(vocab_dict["STRING1"],"").replace(vocab_dict["STRING2"],"").replace(vocab_dict["<topic1>"],"").replace(vocab_dict["<topic2>"],"").replace(vocab_dict["<topic3>"],"").replace(" ","") for s in sss]

    rel_d1={'<https://dblp.org/rdf/schema#authoredBy>': '<authoredBy>', '<https://dblp.org/rdf/schema#wikidata>': '<wikidata>', '<https://dblp.org/rdf/schema#primaryAffiliation>': '<primaryAffiliation>', '<https://dblp.org/rdf/schema#webpage>': '<webpage>', '<https://dblp.org/rdf/schema#yearOfPublication>': '<yearOfPublication>', '<https://dblp.org/rdf/schema#publishedIn>': '<publishedIn>', '<https://dblp.org/rdf/schema#title>': '<title>', '<https://dblp.org/rdf/schema#numberOfCreators>': '<numberOfCreators>'}
    rel_d2={'<https://dblp.org/rdf/schema#wikidata>': '<wikidata>', '<https://dblp.org/rdf/schema#authoredBy>': '<authoredBy>', '<https://dblp.org/rdf/schema#webpage>': '<webpage>', '<https://dblp.org/rdf/schema#primaryAffiliation>': '<primaryAffiliation>', '<https://dblp.org/rdf/schema#orcid>': '<orcid>', '<https://dblp.org/rdf/schema#publishedIn>': '<publishedIn>', '<https://dblp.org/rdf/schema#yearOfPublication>': '<yearOfPublication>', '<https://dblp.org/rdf/schema#title>': '<title>', '<https://dblp.org/rdf/schema#numberOfCreators>': '<numberOfCreators>', '<https://dblp.org/rdf/schema#bibtexType>': '<bibtexType>', '<http://purl.org/dc/terms/bibtexType>': '<bibtexType>', '<http://purl.org/net/nknouf/ns/bibtex#Article>': '<Article>', '<http://purl.org/net/nknouf/ns/bibtex#Inproceedings>': '<Inproceedings>'}
    rel_d3={'<https://dblp.org/rdf/schema#primaryAffiliation>': '<primaryAffiliation>', '<https://dblp.org/rdf/schema#authoredBy>': '<authoredBy>', '<https://dblp.org/rdf/schema#orcid>': '<orcid>', '<https://dblp.org/rdf/schema#webpage>': '<webpage>', '<https://dblp.org/rdf/schema#wikidata>': '<wikidata>', '<https://dblp.org/rdf/schema#publishedIn>': '<publishedIn>', '<https://dblp.org/rdf/schema#yearOfPublication>': '<yearOfPublication>', '<https://dblp.org/rdf/schema#title>': '<title>', '<https://dblp.org/rdf/schema#numberOfCreators>': '<numberOfCreators>', '<https://dblp.org/rdf/schema#bibtexType>': '<bibtexType>', '<http://purl.org/dc/terms/bibtexType>': '<bibtexType>', '<http://purl.org/net/nknouf/ns/bibtex#Inproceedings>': '<Inproceedings>', '<http://purl.org/net/nknouf/ns/bibtex#Article>': '<Article>'}
    rel_d={**rel_d1,**rel_d2,**rel_d3}
    rel_d = {v: [k] for k, v in rel_d.items()}
    rel_d["<bibtexType>"]=rel_d["<bibtexType>"]+["<https://dblp.org/rdf/schema#bibtexType>"]
    
    return sss,sss_s,vocab_dict,rel_d

def get_template(prediction,sss,sss_s,cans=3):
    fnd=re.findall(r'(<.*?>)', prediction)
    tl=[]
    if fnd is not None and len(fnd)>0:
        ppp="".join(list(fnd))
        for i in difflib.get_close_matches(ppp, sss_s, n=cans,cutoff=0.1):
            tl.append(sss[sss_s.index(i)])
    return tl
    
def extract_info(prediction,vocab_dict,verbose=False):
    for i in vocab_dict.values():
        prediction=prediction.replace(i," # ")
    # decoded_string = input_string.encode('utf-8').decode('unicode_escape').encode('utf-8').decode('utf-8')
    prediction=prediction.replace("   "," ").replace("  "," ").strip().encode('utf-8').decode('utf-8')

    phrase = re.findall(r"(?:(?![ #]+)[\w'-\\:/,ÃãÍÇíçÑñÜü ]+ ?)+", prediction)

    if phrase and len(phrase)>0:
        phrase=[p.strip() for p in phrase]
    info_dict={"entity":[],"number":[],"string":[]}
        
    for p in phrase:
        if p.startswith("'") and p.endswith("'"):
            info_dict["string"].append(p)
        elif p.isdigit():
            info_dict["number"].append(p)
        elif len(p)>0:
            info_dict["entity"].append(p)
    mapping={}
    for e in info_dict["entity"]:
        url=get_url(e)
        if len(url)>0:
            mapping[e]=url
        else:
            print(f"NO RESULT FOR {e}!")
    if verbose:
        print(prediction)
        print(info_dict)
        print(mapping)
    return info_dict,mapping
            
def to_query(prediction,sss,sss_s,vocab_dict,rel_d,cans=3):
    candidate_templates=get_template(prediction,sss,sss_s,cans)
    info_dict,mapping=extract_info(prediction,vocab_dict)
    special={"<isnot>":"!=","<dot>":".","?answer <comma_sep>":"?answer; separator=', '","<is_int>":"xsd:integer","<xiaoyu>":"<","<dayu>":">","<comma>":",","<within>":"> YEAR(NOW())-"}
    query_l=[]

    for template in candidate_templates:
        nc=0
        if vocab_dict["<topic3>"] in template:
            # num_entities=3
            nc+=1
        if vocab_dict["<topic2>"] in template:
            nc+=1
        if vocab_dict["<topic1>"] in template:
            nc+=1
        num_entities=nc
        
        if vocab_dict["STRING2"] in template:
            num_string=2
        elif vocab_dict["STRING1"] in template:
            num_string=1
        else:
            num_string=0
        
        if vocab_dict["<num>"] in template:
            num_num=1
        else:
            num_num=0

        if len(mapping)>=num_entities and len(info_dict["string"])>=num_string and len(info_dict["number"])>=num_num:
            for key, value in reversed(OrderedDict(vocab_dict).items()):
                template=template.replace(value, key)
            if num_string == 2:
                if template.index("STRING1")<template.index("STRING2"):
                    template=template.replace("STRING1", info_dict["string"][0])
                    template=template.replace("STRING2", info_dict["string"][1])
                else:
                    template=template.replace("STRING1", info_dict["string"][1])
                    template=template.replace("STRING2", info_dict["string"][0])
            elif num_string == 1:
                template=template.replace("STRING1", info_dict["string"][0])
            if num_num ==1:
                template=template.replace("<num>", info_dict["number"][0])

            tmp=[template,template]
            for k,v in rel_d.items():
                if k in tmp[0]:
                    if len(v)>1:
                        tmp[0]=tmp[0].replace(k,v[0])
                        tmp[1]=tmp[1].replace(k,v[1])
                    else:
                        tmp[0]=tmp[0].replace(k,v[0])
                        tmp[1]=tmp[1].replace(k,v[0])
            if tmp[0]==tmp[1]:
                tmp=list(set(tmp))

            for t in tmp:
                topics = re.findall(r'<topic\d+>', t)
                
                mapp={}
                for topic in topics:
                    mapp[topic]=""

                for i in range(len(mapp.keys())):
                    mapp[list(mapp.keys())[i]]=list(mapping.values())[i]
                
                
                combinations = list(product(*mapp.values()))

                mapp_df = pd.DataFrame(combinations, columns=mapp.keys())

                for k,v in special.items():
                    t=t.replace(k,v)

                cols=list(mapp.keys())
                for i,row in mapp_df.iterrows():
                    query_tmp=t
                    for col in cols:

                        query_tmp=query_tmp.replace(col,"<"+row[col]+">")
                    query_l.append(query_tmp)

    return query_l,info_dict,mapping

def run_pipeline(train,sss,sss_s,vocab_dict,rel_d,verbose=True):

    ll=[]
    l=[]
    info_l=[]
    map_l=[]
    
    if isinstance(train.iloc[0]["question"],dict):
        train["question"]=train["question"].apply(lambda x:x["string"])
    if "query" in train.keys():
        if isinstance(train.iloc[0]["query"],dict):
            train["query"]=train["query"].apply(lambda x:x["sparql"])
    
    for i,row in tqdm(train.iterrows()):
            info,mappinggs=None,None
            ql,info,mappinggs=to_query(row["prediction"],sss,sss_s,vocab_dict,rel_d)
            
            if verbose:
                logger.info(f"-----------------------------Index {i}.......--------------------")
                logger.info(f"question: {row['question']}")
                logger.info(f"extract info: {info}")
                logger.info(f"mappings: {mappinggs}:")
                if "query" in train.keys():
                    logger.info(f"gold query: {row['query']}:")
                
            if ql and len(ql)>0:
                ll.append(ql)
            else:
                ll.append(None)
            info_l.append(info)
            map_l.append(mappinggs)
            qlnew=[q.replace("  ","").replace(" ","") for q in ql]
             
            if verbose:    
                if len(ql)>0:
                    logger.info(f"First potential query:  {ql[0]}")
                else:
                    logger.info(f"NO QUERY !!!!!!!!!!!!!!!!!!!")
                    
            if "query" in train.keys():
                if row["query"].replace("  ","").replace(" ","").encode('utf-8').decode('utf-8') in qlnew:
                    flag=True
                else:
                    flag=False

                if verbose :
                    logger.info(f"Index {i} - test_ok: {flag}")
                
                l.append(flag)
            else:
                l.append(None)

    train["potential_query"]=ll
    if "query" in train.keys():
        train["test_ok"]=l
        print(train["test_ok"].value_counts())
    train["info_dict"]=info_l
    train["mappings"]=map_l
    train.info()

    return train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference using BART')
    parser.add_argument("--default_dir", type=str, default="./",
                        help="default_dir", dest="default_dir")
    parser.add_argument("--input", type=str, help="path for input dataframe",
                        default="inference_heldout.json", dest="input")
    parser.add_argument("--source", type=str, help="input column",
                        default="prediction", dest="source")
    parser.add_argument("--resume_dir", default='logs', type=str)
    parser.add_argument("--save_name", type=str, help="save path for output dataframe saved as json file",
                        default="postprocess_heldout.json",
                        dest="save_name")
    parser.add_argument("--resume_prefix", type=str, default='v1')
    parser.add_argument("--sleep", type=float, default=3, help="sleep time")
    
    args = parser.parse_args()
    print("input args: ", args)

    path=os.path.join(args.default_dir, args.resume_dir, args.resume_prefix, args.input)
    print("loading from path: ", path)
    train=pd.read_json(path)
    train.info()
    
    print("preparing......")
    sss,sss_s,vocab_dict,rel_d=prepare()

    train=run_pipeline(train,sss,sss_s,vocab_dict,rel_d,True)
    
    filename=os.path.join(args.default_dir, args.resume_dir, args.resume_prefix, args.save_name)
    print("saving to path: ", filename)
    train.to_json(filename, orient='records', default_handler=str)
    print("done")