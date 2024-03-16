import json
import pandas as pd
from tqdm import tqdm
import os
import re


def read_datasets(ddd,dirpath="data/DBLP-QuAD"):
    q_path = os.path.join(dirpath, ddd,"questions.json")
    with open(q_path, 'r') as json_file:
        data = json.load(json_file)
    df_q = pd.DataFrame(data["questions"])
    return df_q[["id","question","query_type","query","template_id","entities","relations"]]

def convert_dict():
    initial=["<topic1>","<topic2>","<topic3>"]+["<isnot>","<within>","<num>","<dot>","<dayu>","<xiaoyu>","<comma_sep>","<is_int>","<comma>"]+["<primaryAffiliation>","<yearOfPublication>","<authoredBy>","<numberOfCreators>","<title>","<webpage>","<publishedIn>","<wikidata>","<orcid>","<bibtexType>","<Inproceedings>","<Article>"]
    extra=['?secondanswer', 'GROUP_CONCAT', '?firstanswer', 'separator', 'DISTINCT', '?answer', '?count', 'EXISTS', 'FILTER', 'SELECT', 'STRING1','STRING2', 'BIND','IF', 'COUNT', 'GROUP', 'LIMIT', 'ORDER', 'UNION', 'WHERE', 'DESC','ASC', 'AVG', 'ASK', 'NOT','MAX','MIN','AS', '?x', '?y', '?z', 'BY',"{","}","(",")"]
    vocab=initial+extra
    vocab_dict={}
    for i,text in enumerate(vocab):
        vocab_dict[text]='<eid_'+str(i)+'>'
    return vocab_dict

def convert_number(x):
    question=x["string"]
    numbers = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10"
    }
    for nk,nv in numbers.items():
        if " in the last {} years".format(nk) in question:
            question=question.replace(" in the last {} years".format(nk)," in the last {} years".format(nv))
    return question

def process(ddd,dirpath="data/DBLP-QuAD"):
    df=read_datasets(ddd)
    df["query"]=df["query"].apply(lambda x: x["sparql"])
    #deal with 'the last ? years'
    df["question"]=df["question"].apply(lambda x: convert_number(x))
    vocab_dict=convert_dict()
    # with open(os.path.join(dirpath, f'{ddd}_link2label.json'), 'r') as json_file:
    #     link2label = json.load(json_file)
    
    ll=[]
    lll=[]
    ll_c=[]
    lll_c=[]
    # dl=[]
    dl2=[]
    rel=set()
    d3={}
    for i in tqdm(range(len(df))):
        row=df.iloc[i]
        entities=[]
        query=row["query"].replace("<<http","<http")
        query_template=query
        # d={}
        special_ent=["http://purl.org/net/nknouf/ns/bibtex#Article","http://purl.org/net/nknouf/ns/bibtex#Inproceedings"]
        for e in row["entities"]:
            if e.strip("<").strip(">") not in special_ent:
                if e.startswith("<http"):
                    entities.append(e)
        if len(entities)==1:
            names=["<topic1>"]
        elif len(entities)==2:
            names=["<topic1>","<topic2>"]
        elif len(entities)==3:
            names=["<topic1>","<topic2>","<topic3>"]
        elif len(entities)==0:
            raise Exception("NO entities")
        else:
            raise Exception("more than 3 entities")
        d2={}    
        for i in range(len(entities)):
            d2[entities[i]]=names[i]
        
        for k,v in d2.items():
            query_template=query_template.replace(k,v)
        
        for t in query.split(" "):
            if "<http" in t:
                if "#" in t:
                    rp=t.split("#")[-1].strip(">")
                    query=query.replace(t,"<"+rp+">")
                    query_template=query_template.replace(t,"<"+rp+">")
                    d3[t]="<"+rp+">"
                elif "<http://purl.org/dc/terms/bibtexType>" ==t:
                    rp="bibtexType"
                    query=query.replace(t,"<"+rp+">")
                    query_template=query_template.replace(t,"<"+rp+">")
                    d3[t]="<"+rp+">"
        query=query.replace(" != "," <isnot> ").replace(" . "," <dot> ").replace("?x > ?y","?x <dayu> ?y").replace("?x < ?y","?x <xiaoyu> ?y").replace("?answer; separator=', '","?answer <comma_sep>").replace("xsd:integer(?","<is_int>(?")
        query_template=query_template.replace(" != "," <isnot> ").replace(" . "," <dot> ").replace("?x > ?y","?x <dayu> ?y").replace("?x < ?y","?x <xiaoyu> ?y").replace("?answer; separator=', '","?answer <comma_sep>").replace("xsd:integer(?","<is_int>(?")
        
        query1=re.sub(r' > YEAR\(NOW\(\)\)-(\d+)', r' <within> \1', query)

        query=re.sub(r' > YEAR\(NOW\(\)\)-(\d+)', r' <within> <num>', query_template)
        string_d={}
        string_l=[]
        if "'" in query:
            fnd = re.findall(r"'([^']*)'",query)
            if fnd is not None and len(fnd)>0:
                for one in set(fnd):
                    string_l.append("'{}'".format(one))
        for i in range(len(string_l)):
            string_d[string_l[i]]="STRING{}".format(i+1)
            query=query.replace(string_l[i],"STRING{}".format(i+1))
            query1=query1.replace(string_l[i],"STRING{}".format(i+1))
        query=query.replace(", "," <comma> ")
        query1=query1.replace(", "," <comma> ")
        
        # for e in entities:
        #     link=e.strip("<").strip(">").strip("<").strip(">")
        #     if link in link2label:
        #         label=link2label[link].strip().strip('"').strip("'").strip('.').strip()
        #     else:
        #         if link.startswith("http"):
        #             if link=="http://purl.org/net/nknouf/ns/bibtex#Article":
        #                 label="<Article>"
        #             elif link=="http://purl.org/net/nknouf/ns/bibtex#Inproceedings":
        #                 label="<Inproceedings>"
        #     if not label.startswith("https:"):
        #         query1=query1.replace(e,label)
        #         d[e]=label
        query_tmp=query1
        for k,v in string_d.items():
            query_tmp=query_tmp.replace(v,k)
        ll.append(query_tmp.replace("  "," ").strip())
        lll.append(query.replace("  "," ").strip())
        #do convert
        for k,v in vocab_dict.items():
            query=query.replace(k,v)
            query1=query1.replace(k,v)
        for k,v in string_d.items():
            query1=query1.replace(vocab_dict[v],k)
            
        ll_c.append(query1.replace("  "," ").strip())    
        lll_c.append(query.replace("  "," ").strip())
        
        # dl.append(d)
        dl2.append(string_d)

    df["processed_query"]=ll
    df["processed_query_template"]=lll
    df["processed_query_converted"]=ll_c
    df["processed_query_template_converted"]=lll_c
    
    # df["link2label"]=dl
    df["string_dict"]=dl2
    
    keep_columns=['id', 'question', 'query','processed_query', 'processed_query_template', 'processed_query_converted', 'processed_query_template_converted']
    df=df[keep_columns]
    return df

if __name__ == "__main__":
    train_q=process("train")
    test_q=process("test")
    dev_q=process("valid")
    
    shuffled_train_all=train_q.sample(frac=1, random_state=42)
    shuffled_test_all=test_q.sample(frac=1, random_state=42)
    shuffled_dev_all=dev_q.sample(frac=1, random_state=42)

    shuffled_train_all.to_json("data/DBLP-QuAD/processed_train_new.json",orient = 'records',default_handler=str)
    shuffled_dev_all.to_json("data/DBLP-QuAD/processed_valid_new.json",orient = 'records',default_handler=str)
    shuffled_test_all.to_json("data/DBLP-QuAD/processed_test_new.json",orient = 'records',default_handler=str)