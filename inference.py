from bart import *
import argparse
import numpy as np


initial=["<topic1>","<topic2>","<topic3>"]+["<isnot>","<within>","<num>","<dot>","<dayu>","<xiaoyu>","<comma_sep>","<is_int>","<comma>"]+["<primaryAffiliation>","<yearOfPublication>","<authoredBy>","<numberOfCreators>","<title>","<webpage>","<publishedIn>","<wikidata>","<orcid>","<bibtexType>","<Inproceedings>","<Article>"]
extra=['?secondanswer', 'GROUP_CONCAT', '?firstanswer', 'separator', 'DISTINCT', '?answer', '?count', 'EXISTS', 'FILTER', 'SELECT', 'STRING1','STRING2', 'BIND','IF', 'COUNT', 'GROUP', 'LIMIT', 'ORDER', 'UNION', 'WHERE', 'DESC','ASC', 'AVG', 'ASK', 'NOT','MAX','MIN','AS', '?x', '?y', '?z', 'BY',"{","}","(",")"]
vocab=initial+extra
vocab_dict={}
for i,text in enumerate(vocab):
    vocab_dict[text]='<eid_'+str(i)+'>'


def load_from_checkpoint(model, resume_ckpt):
    print("loading from checkpoint......path: ", resume_ckpt)
    checkpoint = torch.load(resume_ckpt, map_location="cpu")
    epoch = checkpoint["epoch"]
    step = checkpoint["global_step"]
    print(
        f"number of executed epochs of finetuned model(starts from 1): {epoch}\nnumber of executed steps of finetuned model(starts from 0): {step}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("finetuned model loaded!")
    return model

def postprocess(prediction):
    prediction = prediction.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("  ",
                                                                                                " ").strip().lower()
    prediction_postprocessed = prediction

    for k, v in vocab_dict.items():
        prediction_postprocessed = prediction_postprocessed.replace(v, k)
    return prediction, prediction_postprocessed

def store_responses(df, references_bart, filename):
    p_l, pp_l = [], []
    for r in references_bart:
        p, pp = postprocess(r)
        p_l.append(p)
        pp_l.append(pp)
    df['prediction'] = p_l
    df['prediction_postprocessed'] = pp_l
    print("saving to path: ", filename)
    df.to_json(filename, orient='records', default_handler=str)

class myDataset(torch.utils.data.Dataset):
    def __init__(self, responselist, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.responselist = responselist

    def __getitem__(self, idx):
        instance = self.responselist[idx]
        return instance

    def __len__(self):
        return len(self.responselist)

def convert_number(x):
    question=x
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference using BART')
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", dest="resume_ckpt")
    parser.add_argument("--default_dir", type=str, default="./",
                        help="default_dir", dest="default_dir")
    parser.add_argument("--resume_dir", type=str, default='logs', help="directory to load model from checkpoint",
                        dest="resume_dir")
    parser.add_argument("--resume_prefix", type=str, dest="resume_prefix")
    parser.add_argument("--resume_option", type=int, default=1,
                        help="1 means best checkpoint, else means last checkpoint",
                        dest="resume_option")
    # parser.add_argument("--save_dir", type=str,
    #                     default="logs",
    #                     dest="save_dir")
    parser.add_argument("--save_name", type=str, help="save path for output dataframe saved as json file",
                        default="inference_heldout.json",
                        dest="save_name")
    parser.add_argument("--input", type=str, help="path for input dataframe, default test v4", dest="input")
    
    parser.add_argument("--bart_version", type=str, help="bart-base or bart-large", default="bart-base",
                        dest="bart_version")
    parser.add_argument("--device", help="device number, default 0", default=0, type=int, dest="device")
    parser.add_argument("--source", type=str, help="input column",
                        default="question", dest="source")
    
    parser.add_argument("--batch_size", type=int, help="batch size ", default=16, dest="batch_size")
    parser.add_argument("--max_length", type=int, help="max_length when encoding", default=512, dest="max_length")
    parser.add_argument("--max_output_length", type=int, default=1024,
                        help="max_output_length for generation", dest="max_output_length")
    parser.add_argument("--min_output_length", type=int, default=8,
                        help="min_output_l for generation", dest="min_output_length")
    parser.add_argument("--eval_beams", type=int, default=5, help="beam size for inference when testing/validating",
                        dest="eval_beams")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=50,
                        help="no_repeat_ngram_size when generating predictions",
                        dest="no_repeat_ngram_size")
    
    parser.add_argument("--use_convert", action="store_true",dest="use_convert")

    args = parser.parse_args()
    print("input args: ", args)

    directory = os.path.join(args.default_dir, args.resume_dir, args.resume_prefix)
    if args.resume_option == 1:
        for filename in os.listdir(directory):
            if "_savebest_" in filename.lower():
                args.resume_ckpt = os.path.join(directory, filename)
                break
    else:
        for filename in os.listdir(directory):
            if filename.lower().startswith("checkpoint_epoch"):
                args.resume_ckpt = os.path.join(directory, filename)

    if args.resume_ckpt is None:
        raise Exception("PLEASE SPECIFY CHECKPOINT PATH!")

    if args.use_convert:
        special_tokens = list(vocab_dict.values())
    else:
        special_tokens = vocab
    # load model
    print("loading bart tokenizer and model......")
    tokenizer = BartTokenizer.from_pretrained("facebook/" + args.bart_version, add_prefix_space=True,
                                              additional_special_tokens=special_tokens)
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/" + args.bart_version)
    bart_model.resize_token_embeddings(len(tokenizer))
    model = load_from_checkpoint(bart_model, args.resume_ckpt)

    # to device
    rank = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    print(f"device: {rank}")
    model.to(rank)
    model.eval()

    # read data
    df_path = args.default_dir +args.input
    print(f"reading dataset from path {df_path}")
    df = pd.read_json(df_path)
    df.info()
    
    if args.source in df.keys():
        if isinstance(df.iloc[0][args.source],dict):
            df[args.source]=df[args.source].apply(lambda x:x["string"])
    else:
        raise Exception(f"{args.source} NOT IN FILE!!!")
        
    df[args.source]=df[args.source].apply(lambda x: convert_number(x))
    
    sourcelist = df[args.source].tolist()

    mydataset = myDataset(sourcelist, tokenizer)
    batch_size = int(args.batch_size)
    mydataloader = torch.utils.data.DataLoader(mydataset, batch_size)

    predictions_l = []
    for step_index, batch_data in tqdm(enumerate(mydataloader), f"[GENERATE]", total=len(mydataloader), mininterval=60):
        texts = batch_data
        encoding = tokenizer(texts, max_length=int(args.max_length), return_tensors='pt', truncation=True,
                             add_prefix_space=True, padding="max_length")
        encoding.to(rank)
        with torch.no_grad():
            output = model.generate(
                **encoding,
                use_cache=True,
                num_beams=args.eval_beams,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=tokenizer.pad_token_id,
                max_length=args.max_output_length,
                min_length=args.min_output_length,
                early_stopping=True,
                no_repeat_ngram_size=args.no_repeat_ngram_size
            )
            predictions = tokenizer.batch_decode(output.tolist(), skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=True)
            predictions_l += predictions
    print("\nlength of predictions: {}".format(len(predictions_l)))
    predictions_l = list(np.array(predictions_l).flatten())

    # store results
    filename = os.path.join(args.default_dir, args.resume_dir, args.resume_prefix, args.save_name)
    store_responses(df, predictions_l, filename)
