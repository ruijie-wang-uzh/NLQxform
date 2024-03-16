import os

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
from tqdm import tqdm
import logging
from transformers.models.bart.modeling_bart import shift_tokens_right
import datetime
import traceback
import json
from bart_utils import get_eval_scores, optimizer_to

initial=["<topic1>","<topic2>","<topic3>"]+["<isnot>","<within>","<num>","<dot>","<dayu>","<xiaoyu>","<comma_sep>","<is_int>","<comma>"]+["<primaryAffiliation>","<yearOfPublication>","<authoredBy>","<numberOfCreators>","<title>","<webpage>","<publishedIn>","<wikidata>","<orcid>","<bibtexType>","<Inproceedings>","<Article>"]
extra=['?secondanswer', 'GROUP_CONCAT', '?firstanswer', 'separator', 'DISTINCT', '?answer', '?count', 'EXISTS', 'FILTER', 'SELECT', 'STRING1','STRING2', 'BIND','IF', 'COUNT', 'GROUP', 'LIMIT', 'ORDER', 'UNION', 'WHERE', 'DESC','ASC', 'AVG', 'ASK', 'NOT','MAX','MIN','AS', '?x', '?y', '?z', 'BY',"{","}","(",")"]
vocab=initial+extra
vocab_dict={}
for i,text in enumerate(vocab):
    vocab_dict[text]='<eid_'+str(i)+'>'

class MyModel(torch.nn.Module):

    def __init__(self, params, tokenizer=None, bart_model=None, logger=None):
        torch.nn.Module.__init__(self)
        self.args = params
        self.args.train_path = os.path.join(self.args.default_dir, self.args.train_path)
        self.args.val_path = os.path.join(self.args.default_dir, self.args.val_path)
        self.args.test_path = os.path.join(self.args.default_dir, self.args.test_path)
        self.args.save_dir = os.path.join(self.args.default_dir, self.args.save_dir)

        self.optimizer = None
        self.scheduler = None

        self.lr_mode = 'min'
        if self.args.early_stopping_metric != 'vloss':
            self.lr_mode = 'max'

        self.history = dict()
        self.track_history_path = os.path.join(self.args.save_dir, self.args.save_prefix, "history_now.json")
        os.makedirs(os.path.dirname(self.track_history_path), exist_ok=True)
        
        if "_converted" in self.args.target:
            self.special_tokens = list(vocab_dict.values())
        else:
            self.special_tokens = vocab
            
        if tokenizer is None or bart_model is None:
            print("loading bart tokenizer and model......")
            self.tokenizer = BartTokenizer.from_pretrained("facebook/" + self.args.bart_version,
                                                               add_prefix_space=True,
                                                               additional_special_tokens=self.special_tokens)
            self.model = BartForConditionalGeneration.from_pretrained("facebook/" + self.args.bart_version)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer=tokenizer
            self.model=bart_model

        if logger is None:
            logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        ids = [0]
        if "," not in str(self.args.gpus):
            if self.args.device is None:
                print("default self.args.device —— cuda:0")
                self.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print('use single gpu: {}'.format(self.args.device))
        else:
            ids = str(self.args.gpus).split(",")
            ids = [int(id.strip()) for id in ids]
            default_gpus = [5, 4, 3, 2, 1, 0]
            for id in ids:
                if id not in default_gpus:
                    raise Exception("INVALID args.gpus!")
            ids = sorted(ids, reverse=True)
            print('use gpus: {}'.format(ids))
            self.args.device = torch.device("cuda:" + str(ids[0]) if torch.cuda.is_available() else "cpu")

        if self.args.resume_ckpt is not None:
            print("RESUME")
            result = self.load_from_checkpoint()
            self._prepare()
            self.configure_optimizers(result)
        else:
            print("NO RESUME!")
            self._prepare()
            self.configure_optimizers()

        self.decoder_start_token_id = self.model.model.config.decoder_start_token_id
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self._get_dataloader()
        self.current_checkpoint = 0
        self.best_checkpoint = None
        self.best_metric = 10000 if self.args.early_stopping_metric == 'vloss' else 0  ## keep track of best dev value of whatever metric is used in early stopping callback
        print(f"initial self.best_metric = {self.best_metric}")

        self.global_step = 0

        self.para_model = None
        if "," not in str(self.args.gpus):
            self.model.to(self.args.device)
        else:
            self.model.to(self.args.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=ids).to(self.args.device)

        print("READY!")

    def _prepare(self):
        if self.args.freeze_option >= 0:
            if self.args.freeze_embeds:
                print("freeze embedding from beginning")
                self.freeze_embeds()
            if self.args.freeze_encoder:
                print("freeze encoder from beginning")
                self.freeze_encoder()

        self.config = self.model.config
        self.config.attention_dropout = self.args.attention_dropout
        self.config.dropout = self.args.dropout
        self.config.activation_dropout = self.args.activation_dropout
        self.config.gradient_checkpointing = self.args.gradient_checkpointing

    def freeze_embeds(self, unfreeze=False):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            self.freeze_params(self.model.module.model.shared)
            for d in [self.model.module.model.encoder, self.model.module.model.decoder]:
                self.freeze_params(d.embed_positions, unfreeze)
                self.freeze_params(d.embed_tokens, unfreeze)
        else:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions, unfreeze)
                self.freeze_params(d.embed_tokens, unfreeze)

    def freeze_encoder(self, unfreeze=False):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            self.freeze_params(self.model.module.get_encoder(), unfreeze)
        else:
            self.freeze_params(self.model.get_encoder(), unfreeze)

    def freeze_params(self, model, unfreeze=False):
        if unfreeze:
            for layer in model.parameters():
                print(f'Unfreezing parameters with shape: {layer.shape}')
                layer.requires_grad = True
        else:
            for layer in model.parameters():
                print(f'Freezing parameters with shape: {layer.shape}')
                layer.requires_grad = False

    def get_features(self, df, pad_to_max_length=True, return_tensors="pt"):
        inputs = [s for s in df[self.args.source]]
        targets = [t for t in df[self.args.target]]
        input_encodings = self.tokenizer(inputs, max_length=self.args.max_length, truncation=True,
                                         padding="max_length" if pad_to_max_length else None,
                                         return_tensors=return_tensors,
                                         add_prefix_space=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(targets, max_length=self.args.max_length, truncation=True,
                                              padding="max_length" if pad_to_max_length else None,
                                              return_tensors=return_tensors,
                                              add_prefix_space=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]}

    def loading_df(self, path: str):
        data = json.load(open(path))
        return pd.json_normalize(data)

    def _get_dataloader(self):
        print("reading data from json file......")
        train = self.loading_df(self.args.train_path)[[self.args.source, self.args.target]]
        val = self.loading_df(self.args.val_path)[[self.args.source, self.args.target]]
        test = self.loading_df(self.args.test_path)[[self.args.source, self.args.target]]
        if self.args.small_dataset:
            train = train[:400]
            val = val[:40]
            test = test[:40]

        print(f"length of train: {len(train)}\nlength of val: {len(val)}\nlength of test: {len(test)}")
        print("getting features......")
        train_f = self.get_features(train)
        val_f = self.get_features(val)
        test_f = self.get_features(test)
        print("preparing dataloaders......")
        train_d = TensorDataset(train_f['input_ids'], train_f['attention_mask'], train_f['labels'])
        val_d = TensorDataset(val_f['input_ids'], val_f['attention_mask'], val_f['labels'])
        test_d = TensorDataset(test_f['input_ids'], test_f['attention_mask'], test_f['labels'])

        self.train_dataloader = DataLoader(train_d, sampler=RandomSampler(train_d), batch_size=self.args.batch_size)
        self.val_dataloader = DataLoader(val_d, batch_size=self.args.batch_size)
        self.test_dataloader = DataLoader(test_d, batch_size=self.args.batch_size)

    def forward(self, batch, current_epoch=None):
        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]

        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id, self.decoder_start_token_id)

        outputs = self.model(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        labels = tgt_ids.clone() 

        if self.args.label_smoothing == 0:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        return loss

    def train(self, num_epochs=None):
        print("*************start training****************")
        print("current self.global_step = ", self.global_step)

        if num_epochs is None:
            num_epochs = self.args.max_epochs

        stop_counter = -1
        last_flag = False
        now_epoch = 0
        pbar = tqdm(range(num_epochs))
        try:
            for epoch in pbar:
                pbar.set_description(f"Epoch  {epoch + 1}")
                now_epoch = epoch
                self.model.train()
                total_loss = 0
                total_celoss = 0
                self.optimizer.zero_grad()
                current_lr = self.optimizer.param_groups[0]['lr']
                print("current learning rate: ", current_lr)

                self.history[f"epoch {epoch + 1}"] = {"train-epoch-end-avg-loss": [], "val-epoch-end-avg-metrics": []}

                batch_idx = 0
                batch_interval = 0
                for batch in tqdm(self.train_dataloader, mininterval=self.args.progress_bar_refresh_interval,
                                  desc="Train"):
                    batch = [
                        t.to(self.args.device) if t is not None else None for t in batch]
                    loss = self.training_step(batch, now_epoch)
                    loss = loss.float().mean().type_as(loss)
                    self.global_step += 1

                    if batch_interval % int(self.args.logger_batch_interval) == 0:
                        self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx + 1}/{len(self.train_dataloader)}] - train-loss: {loss}")
                    batch_idx += 1
                    batch_interval += 1

                    total_loss += loss.item()

                avg_loss = total_loss / len(self.train_dataloader)
                self.history[f"epoch {epoch + 1}"]["train-epoch-end-avg-loss"] = [avg_loss]
                print(f"Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {str(avg_loss)}")
                
                if epoch%4==3:
                    self.model.eval()
                    scores_l = []
                    batch_idx = 0
                    batch_interval = 0
                    total_celoss = 0
                    with torch.no_grad():
                        for batch in tqdm(self.val_dataloader, mininterval=self.args.progress_bar_refresh_interval,
                                          desc="Validation"):
                            batch = [
                                t.to(self.args.device) if t is not None else None for t in batch]
                            scores,vloss = self.validation_step(batch, now_epoch)
                            if batch_interval % int(self.args.logger_batch_interval) == 0:
                                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx + 1}/{len(self.val_dataloader)}] - metrics: {str(scores)}")
                            batch_idx += 1
                            batch_interval += 1
                            total_celoss += vloss.item()
                            scores_l.append(scores)

                    avg_celoss = total_celoss / len(self.val_dataloader)
                    flag, metric_logs = self.validation_epoch_end(scores_l, now_epoch)
                    simplified_metric_logs = {}
                    for key, value in metric_logs.items():
                        simplified_metric_logs[key] = value.item()

                    self.history[f"epoch {epoch + 1}"]["val-epoch-end-avg-metrics"] = [simplified_metric_logs,avg_celoss]
                else:
                    flag=True
                    self.history[f"epoch {epoch + 1}"]["val-epoch-end-avg-metrics"] = [None,None]

                with open(self.track_history_path, "w") as fp:
                    json.dump(self.history, fp)

                if self.args.save_best and flag:
                    self.save_model(now_epoch, best=True)

                if not flag:
                    if last_flag:
                        stop_counter = 0
                    stop_counter += 1

                    if stop_counter >= self.args.patience:
                        print("Early stopping triggered.")
                        break
                else:
                    stop_counter = 0
                print(f"number of continuous epochs not improved: {stop_counter}")
                last_flag = flag

            print("current self.global_step = ", self.global_step)
            print("*************training done****************")
        except Exception as e:
            print("*****ERROR OCCURRED!!!!!STOP TRAINING*****")
            print("error info: ", e)
            traceback.print_exc()
        finally:
            self.save_model(now_epoch)

    def training_step(self, batch, now_epoch):
        loss = self.forward(batch, now_epoch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def postprocess(self, strl):
        new_l = []
        for one in strl:
            new_one = one.replace("<s>", "").replace("<pad>", "").replace("</s>", "").strip()
            if "_converted" in self.args.target:
                for k, v in vocab_dict.items():
                    new_one = new_one.replace(v, k)
            new_l.append(new_one)
        return new_l

    def validation_step(self, batch, now_epoch):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            for p in self.model.module.parameters():
                p.requires_grad = False
        else:
            for p in self.model.parameters():
                p.requires_grad = False

        vloss = self.forward(batch, now_epoch)

        vloss = vloss.float().mean().type_as(vloss)
        
        input_ids, attention_mask, output_ids = batch
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            generated_ids = self.model.module.generate(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                max_length=self.args.max_output_length, min_length=self.args.min_output_length,
                num_beams=self.args.eval_beams,
                pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.pad_token_id,
                early_stopping=True,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size
            )
        else:
            generated_ids = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                max_length=self.args.max_output_length, min_length=self.args.min_output_length,
                num_beams=self.args.eval_beams,
                pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.pad_token_id,
                early_stopping=True,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size
            )

        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=True)

        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=False,
                                               clean_up_tokenization_spaces=True)

        generated_str_processed = self.postprocess(generated_str)
        gold_str_processed = self.postprocess(gold_str)

        scores = get_eval_scores(gold_str_processed, generated_str_processed, vloss)

        outfile = self.args.save_dir + "/" + self.args.save_prefix + "/_val_out_checkpoint_" + str(
            self.current_checkpoint)
        outfile_processed = self.args.save_dir + "/" + self.args.save_prefix + "/_val_out_PROCESSED_checkpoint_" + str(
            self.current_checkpoint)
        outfile_gold = self.args.save_dir + "/" + self.args.save_prefix + "/_gold_query_PROCESSED"

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        os.makedirs(os.path.dirname(outfile_processed), exist_ok=True)
        os.makedirs(os.path.dirname(outfile_gold), exist_ok=True)
        with open(outfile, 'a') as f:
            for sample in generated_str:
                f.write(sample + "\n")
        with open(outfile_processed, 'a') as f:
            for sample in generated_str_processed:
                f.write(sample + "\n")
        with open(outfile_gold, 'a') as f:
            for sample in gold_str_processed:
                f.write(sample + "\n")

        return scores,vloss

    def validation_epoch_end(self, outputs, current_epoch=None):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            for p in self.model.module.parameters():
                p.requires_grad = True
        else:
            for p in self.model.parameters():
                p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        print("Evaluation on checkpoint [{}] ".format(self.current_checkpoint))
        print(logs)

        self.scheduler.patience = self.args.lr_reduce_patience

        self.scheduler.step(logs[self.args.early_stopping_metric])

        flag = False
        if self.args.early_stopping_metric == 'vloss':
            if logs['vloss'] < self.best_metric:
                self.best_checkpoint = self.current_checkpoint
                flag = True
                print("New best checkpoint {}, with {} {}<{}.".format(
                    self.best_checkpoint, self.args.early_stopping_metric, logs['vloss'], self.best_metric)
                )
                self.best_metric = logs['vloss']
            else:
                print("score of checkpoint {} not improved, with {} {}>={}.".format(
                    self.current_checkpoint, self.args.early_stopping_metric, logs['vloss'], self.best_metric)
                )
        else:
            if logs[self.args.early_stopping_metric] > self.best_metric:
                self.best_checkpoint = self.current_checkpoint
                flag = True
                print("New best checkpoint {}, with {} {}>{}.".format(self.best_checkpoint,
                                                                      self.args.early_stopping_metric,
                                                                      logs[self.args.early_stopping_metric],
                                                                      self.best_metric))
                self.best_metric = logs[self.args.early_stopping_metric]
            else:
                print("score of checkpoint {} not improved, with {} {}<={}.".format(self.current_checkpoint,
                                                                                    self.args.early_stopping_metric,
                                                                                    logs[
                                                                                        self.args.early_stopping_metric],
                                                                                    self.best_metric))
        self.current_checkpoint += 1
        return flag, logs

    def configure_optimizers(self, result=None):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if result is not None:
            self.optimizer.load_state_dict(result[0])
            optimizer_to(self.optimizer, self.args.device)
            if not result[1]:
                self.optimizer.param_groups[0]['lr'] = self.args.learning_rate

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode=self.lr_mode,
                                                                    factor=self.args.lr_reduce_factor,
                                                                    patience=self.args.lr_reduce_patience,
                                                                    cooldown=self.args.cooldown,
                                                                    min_lr=self.args.min_lr,
                                                                    verbose=True)
        self.logger.info(
            f'set reduce LR on plateau schedule with mode={self.lr_mode}, factor={self.args.lr_reduce_factor} and patience={self.args.lr_reduce_patience}.')

    def save_model(self, current_epoch, best=False):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model
        current_lr = self.optimizer.param_groups[0]['lr']
        if isinstance(self.optimizer, torch.nn.DataParallel) or isinstance(self.optimizer,
                                                                           torch.nn.parallel.DistributedDataParallel):
            optimizer = self.optimizer.module
        else:
            optimizer = self.optimizer

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': current_epoch + 1,
            'global_step': self.global_step,
            'latest_lr': current_lr
        }
        his_path = "history.json"
        if best:
            custom_checkpoint_path = f"checkpoint_SAVEBEST_metric_{self.args.early_stopping_metric}.ckpt"
        else:
            custom_checkpoint_path = f"checkpoint_epoch:{current_epoch + 1}_step:{self.global_step}_metric:{self.args.early_stopping_metric}_time:{datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')}.ckpt"
            his_path = os.path.join(self.args.save_dir, self.args.save_prefix, his_path)
            os.makedirs(os.path.dirname(his_path), exist_ok=True)

        path = os.path.join(self.args.save_dir, self.args.save_prefix, custom_checkpoint_path)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not best:
            print(f'saving latest model to path: {path}')
            torch.save(checkpoint, path)
            print(f'saving history to path: {his_path}')
            with open(his_path, "w") as fp:
                json.dump(self.history, fp)
            print("saving done!")
        else:
            print(f"updating best model to best checkpoint {self.best_checkpoint}")
            torch.save(checkpoint, path)

    def load_from_checkpoint(self):
        print("loading from checkpoint......path: ", self.args.resume_ckpt)
        checkpoint = torch.load(self.args.resume_ckpt, map_location=self.args.device)
        epoch = checkpoint["epoch"]
        step = checkpoint["global_step"]
        last_lr = checkpoint["latest_lr"]
        print(
            f"number of executed epochs of finetuned model(starts from 1): {epoch}\nnumber of executed steps of finetuned model(starts from 0): {step}")
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        print("finetuned model loaded!")
        if self.args.optim_resume:
            if self.args.lr_resume:
                print(f"loading optimizer from checkpoint, and learning rate is {last_lr}")
                return [checkpoint["optimizer_state_dict"], True]
            else:
                print(
                    f"loading optimizer from checkpoint, but updating learning rate from {last_lr} to argument learning_rate {self.args.learning_rate}")
                return [checkpoint["optimizer_state_dict"], False]
        else:
            print(
                "optimizer from checkpoint is NOT loaded, and optimizer will be initialized using argument learning_rate")
            return None