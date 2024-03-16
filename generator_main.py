import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from bart import *
from inference import *
from postprocess import *
from do_query import *
import argparse
import numpy as np
import time
from pprint import pprint as pp
import colorama
from colorama import Fore, Style
import ssl

class Generator:
    def __init__(self, args):
        self.args = args
        self.load()
        self.generate(self.args.verbose,self.args.no_color)

    def load(self):
        self.sss,self.sss_s,self.vocab_dict,self.rel_d=self.prepare()
        print("loading model and tokenizer")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/" + self.args.bart_version, add_prefix_space=True,
                                              additional_special_tokens=list(self.vocab_dict.values()))
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/" + self.args.bart_version)
        bart_model.resize_token_embeddings(len(self.tokenizer))
        self.model = load_from_checkpoint(bart_model, self.args.resume_ckpt)

        self.rank = torch.device("cuda:" + str(self.args.device) if torch.cuda.is_available() else "cpu")
        print(f"device: {self.rank}")
        self.model.to(self.rank)
        self.model.eval()
        print("loading done!")
        
    def do_inference(self,question,verbose,no_color):
        encoding = self.tokenizer(question, max_length=int(self.args.max_length), return_tensors='pt', truncation=True,
                             add_prefix_space=True, padding="max_length")
        encoding.to(self.rank)
        with torch.no_grad():
            output = self.model.generate(
                **encoding,
                use_cache=True,
                num_beams=self.args.eval_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=self.tokenizer.pad_token_id,
                max_length=self.args.max_output_length,
                min_length=self.args.min_output_length,
                early_stopping=True,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size
            )
            prediction = self.tokenizer.decode(output[0], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=True)
            prediction = prediction.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("  ",
                                                                                                " ").strip().lower()
            prediction_postprocessed = prediction
            for k, v in self.vocab_dict.items():
                prediction_postprocessed = prediction_postprocessed.replace(v, k)
        if verbose:
            if no_color:
                print("**************************PROCESSING**************************")
            else:
                print(Fore.LIGHTMAGENTA_EX+"**************************PROCESSING**************************")
            print(f"prediction: {prediction}\nafter replacing <eid_?>: {prediction_postprocessed}\n")
        return prediction
    
    def convert_number(self,question):
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
    
    def prepare(self):
        # data=pd.read_json("../file/DBLP-QuAD/processed_train_new.json")
        # template_s=set(list(data["processed_query_template_converted"]))
        # data=pd.read_json("../file/DBLP-QuAD/processed_test_new.json")
        # template_s2=set(list(data["processed_query_template_converted"]))
        # data=pd.read_json("../file/DBLP-QuAD/processed_valid_new.json")
        # template_s3=set(list(data["processed_query_template_converted"]))
        # sss=template_s.union(template_s2).union(template_s3)
        # sss=list(sss)
        sss=['<eid_33> <eid_28> <eid_50><eid_58><eid_10><eid_58><eid_29><eid_59><eid_59> <eid_51> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57> <eid_42> <eid_56> <eid_29> <eid_14> <eid_1> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57> <eid_57>', '<eid_33> <eid_29> <eid_43> <eid_56> <eid_0> <eid_15> <eid_29> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_13> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_0> <eid_13> <eid_34> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_6> <eid_52> <eid_14> <eid_26> <eid_32><eid_58><eid_26> <eid_3> <eid_0><eid_59> <eid_6> <eid_26> <eid_12> <eid_24> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_20> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_14> <eid_26> <eid_6> <eid_26> <eid_12> <eid_24> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_29> <eid_57>', '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_29> <eid_32><eid_58><eid_29> <eid_3> <eid_0><eid_59> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_34> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_53> <eid_14> <eid_52> <eid_32> <eid_58><eid_53> <eid_3> <eid_0><eid_59> <eid_6> <eid_53> <eid_18> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>', '<eid_47> <eid_56> <eid_2> <eid_14> <eid_0> <eid_6> <eid_2> <eid_14> <eid_1> <eid_57>', '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1', '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_29> <eid_57>', '<eid_33> <eid_58><eid_46><eid_58><eid_30><eid_59> <eid_51> <eid_29><eid_59> <eid_56> <eid_33> <eid_58><eid_38><eid_58><eid_53><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_57> <eid_39> <eid_55> <eid_53> <eid_57>', '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_53> <eid_14> <eid_52> <eid_6> <eid_53> <eid_18> <eid_29> <eid_32> <eid_58><eid_53> <eid_3> <eid_0><eid_59> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_14> <eid_26> <eid_6> <eid_52> <eid_14> <eid_26> <eid_6> <eid_52> <eid_18> <eid_24> <eid_32> <eid_58><eid_52> <eid_3> <eid_0><eid_59> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_20> <eid_52> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_53> <eid_14> <eid_52> <eid_32> <eid_58><eid_53> <eid_3> <eid_0><eid_59> <eid_6> <eid_53> <eid_18> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_0> <eid_18> <eid_34> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_49><eid_58><eid_10><eid_58><eid_29><eid_59><eid_59> <eid_51> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57>', '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_34> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_0> <eid_18> <eid_34> <eid_6> <eid_0> <eid_13> <eid_35> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_15> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_15> <eid_29> <eid_57> <eid_57>', '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_29> <eid_32><eid_58><eid_29> <eid_3> <eid_0><eid_59><eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1', '<eid_33> <eid_28> <eid_29> <eid_50><eid_58><eid_10><eid_58><eid_53><eid_59><eid_59> <eid_51> <eid_53> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_52> <eid_14> <eid_29> <eid_6> <eid_52> <eid_13> <eid_53> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_26> <eid_14> <eid_0> <eid_6> <eid_26> <eid_13> <eid_24> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_13> <eid_29> <eid_57>', '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_21> <eid_23> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_13> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_56> <eid_29> <eid_18> <eid_35> <eid_57> <eid_42> <eid_56> <eid_29> <eid_18> <eid_34> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_56> <eid_29> <eid_18> <eid_34> <eid_57> <eid_42> <eid_56> <eid_29> <eid_18> <eid_35> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_13> <eid_52> <eid_6> <eid_1> <eid_13> <eid_53> <eid_6> <eid_36><eid_58><eid_37><eid_58><eid_52> <eid_8> <eid_53> <eid_11> <eid_0> <eid_11> <eid_1><eid_59> <eid_51> <eid_29><eid_59> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_12> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_12> <eid_29> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_15> <eid_52> <eid_6> <eid_1> <eid_15> <eid_53> <eid_6> <eid_36><eid_58><eid_37><eid_58><eid_52> <eid_7> <eid_53> <eid_11> <eid_0> <eid_11> <https://dblp.org/rec/conf/sigmetrics/GastH18><eid_59> <eid_51> <eid_29><eid_59> <eid_57>', '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_29> <eid_32><eid_58><eid_29> <eid_3> <eid_0><eid_59> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_18> <eid_35> <eid_6> <eid_0> <eid_13> <eid_34> <eid_6> <eid_0> <eid_16> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_17> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_1> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_13> <eid_53> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_57>', '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_54> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_53> <eid_14> <eid_29> <eid_6> <eid_53> <eid_13> <eid_54> <eid_57> <eid_39> <eid_55> <eid_54> <eid_57> <eid_41> <eid_55> <eid_45><eid_58><eid_54><eid_59> <eid_40> 1', '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_32> <eid_48> <eid_31> <eid_56> <eid_1> <eid_14> <eid_0> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_18> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_18> <eid_34> <eid_6> <eid_0> <eid_13> <eid_35> <eid_6> <eid_0> <eid_16> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_14> <eid_26> <eid_6> <eid_24> <eid_14> <eid_26> <eid_32> <eid_58><eid_24> <eid_3> <eid_0><eid_59> <eid_57>', '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_0> <eid_18> <eid_35> <eid_6> <eid_0> <eid_13> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_34> <eid_6> <eid_0> <eid_13> <eid_29> <eid_57>', '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_32> <eid_58><eid_29> <eid_3> <eid_0><eid_59> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_14> <eid_29> <eid_57> <eid_42> <eid_56> <eid_0> <eid_14> <eid_29> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_26> <eid_32><eid_58><eid_26> <eid_3> <eid_0><eid_59> <eid_6> <eid_26> <eid_12> <eid_24> <eid_57>', '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_21> <eid_22> <eid_57>', '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_52><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_52> <eid_14> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1', '<eid_47> <eid_56> <eid_2> <eid_14> <eid_0> <eid_6> <eid_2> <eid_14> <eid_1> <eid_32> <eid_48> <eid_31> <eid_56> <eid_2> <eid_14> <eid_0> <eid_6> <eid_2> <eid_14> <eid_1> <eid_57> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_1> <eid_14> <eid_52> <eid_32> <eid_58><eid_1> <eid_3> <eid_0><eid_59> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_1> <eid_57>', '<eid_33> <eid_28> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_6> <eid_29> <eid_13> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_26> <eid_6> <eid_52> <eid_16> <eid_24> <eid_57>', '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_0> <eid_13> <eid_53> <eid_6> <eid_54> <eid_14> <eid_52> <eid_6> <eid_54> <eid_13> <eid_29> <eid_32> <eid_58><eid_29> <eid_3> <eid_53><eid_59> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_12> <eid_34> <eid_57>', '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_1> <eid_6> <eid_0> <eid_18> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_17> <eid_29> <eid_57>', '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_1> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_12> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_18> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_18> <eid_29> <eid_57> <eid_57>', '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_53> <eid_6> <eid_53> <eid_12> <eid_34> <eid_57>', '<eid_33> <eid_58><eid_46><eid_58><eid_30><eid_59> <eid_51> <eid_29><eid_59> <eid_56> <eid_33> <eid_58><eid_38><eid_58><eid_53><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_15> <eid_53> <eid_57> <eid_39> <eid_55> <eid_53> <eid_57>', '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_1> <eid_14> <eid_52> <eid_32> <eid_58><eid_1> <eid_3> <eid_0><eid_59> <eid_32> <eid_48> <eid_31> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_1> <eid_14> <eid_52> <eid_32> <eid_58><eid_1> <eid_3> <eid_0><eid_59> <eid_57> <eid_57>', '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_29> <eid_12> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_13> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_13> <eid_29> <eid_57> <eid_57>', '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_1> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_15> <eid_52> <eid_6> <eid_1> <eid_15> <eid_53> <eid_6> <eid_36><eid_58><eid_37><eid_58><eid_52> <eid_7> <eid_53> <eid_11> <eid_0> <eid_11> <eid_1><eid_59> <eid_51> <eid_29><eid_59> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_19> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_6> <eid_52> <eid_18> <eid_26> <eid_6> <eid_52> <eid_16> <eid_24> <eid_57>', '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_34> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_29> <eid_14> <eid_0> <eid_57> <eid_42> <eid_56> <eid_29> <eid_14> <eid_1> <eid_57> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_21> <eid_29> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_20> <eid_29> <eid_57>', '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_45><eid_58><eid_30><eid_59> <eid_40> 1', '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_18> <eid_26> <eid_6> <eid_0> <eid_13> <eid_24> <eid_57>', '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_6> <eid_52> <eid_18> <eid_34> <eid_57>', '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>']

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

    def generate(self,verbose=True,no_color=False):
        print("INPUT 'BYE' IF YOU WANT TO STOP AND EXIT!")
        while True:
            if no_color:
                user_input = input("\nplease input your question: \n") 
            else:
                user_input = input(Fore.BLUE+"\nplease input your question: \n")
                print(Fore.YELLOW+"",end="")
            start_t=time.time()
            if user_input.strip()=="BYE":
                break
            else:
                prediction=self.do_inference(self.convert_number(user_input),verbose,no_color)
                ql,info,mappinggs=to_query(prediction,self.sss,self.sss_s,self.vocab_dict,self.rel_d)
                if verbose:
                    print("information extracted from prediction:")
                    pp(info,indent=4,width=200,depth=2)
                    print("entity label-link mappings:")
                    pp(mappinggs,indent=4,width=200,depth=2)
                    print("potential queries:")
                    pp(ql,depth=1,indent=4,width=200)
                simplified=[]
                if ql and len(ql)>0:
                    for query in ql:
                        if verbose:
                            print("\nquerying: "+query)
                        # result=do_query(query)
                        sparql.setQuery(f"""{query}""")
                        sparql.setReturnFormat(JSON)
                        result = sparql.query().convert()

                        if "results" in result.keys():
                            if len(result["results"]["bindings"])!=0:
                                simplified=get_answer(result)
                                if len(simplified)>0:
                                    break
                        elif "boolean" in result.keys():
                            simplified=get_answer(result)
                            if len(simplified)>0:
                                break
                        if verbose:
                            print("no valid answer for this query.")
                
                if verbose:
                    end_t=time.time()
                    print(f"\nit took {round(end_t-start_t,4)} seconds to process this question.")
                    print("*****************************DONE*****************************")
                
                if len(simplified)!=0:
                    if no_color:
                        print("final answer:")
                    else:
                        print(Fore.GREEN+"final answer:")
                    pp(simplified,indent=4)
                else:
                    if no_color:
                        print("no result for this question, please try another one.")
                    else:
                        print(Fore.RED+"no result for this question, please try another one.")
                        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", dest="resume_ckpt")
    parser.add_argument("--default_dir", type=str, default="./",
                        help="default_dir", dest="default_dir")
    parser.add_argument("--resume_dir", type=str, default='logs', help="directory to load model from checkpoint",
                        dest="resume_dir")
    parser.add_argument("--resume_prefix", type=str, default="v2", dest="resume_prefix")
    parser.add_argument("--resume_option", type=int, default=1,
                        help="1 means best checkpoint, else means last checkpoint",
                        dest="resume_option")
    parser.add_argument("--bart_version", type=str, help="bart-base or bart-large", default="bart-base",
                        dest="bart_version")
    parser.add_argument("--device", help="device number, default 0", default=0, type=int, dest="device")
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
    parser.add_argument("--verbose", action="store_true",dest="verbose")
    parser.add_argument("--no_color", action="store_true",dest="no_color")
    parser.add_argument("--sparql_endpoint", default="https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql", type=str)
    
    args = parser.parse_args()

    ssl._create_default_https_context = ssl._create_unverified_context
    sparql = SPARQLWrapper(args.sparql_endpoint)

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

    nlqxform=Generator(args)
    
