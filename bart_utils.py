import torch
import sacrebleu
from rouge_score import rouge_scorer


def get_eval_scores(gold_strs, generated_strs, vloss=None):
    if vloss is None:
        vloss = torch.zeros(len(gold_strs))
    scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
    rouge1 = rouge2 = rougel = rougelsum = 0.0
    for ref, pred in zip(gold_strs, generated_strs):
        score = scorer.score(ref, pred)
        rouge1 += score['rouge1'].fmeasure
        rouge2 += score['rouge2'].fmeasure
        rougel += score['rougeL'].fmeasure
        rougelsum += score['rougeLsum'].fmeasure
    rouge1 /= len(generated_strs)
    rouge2 /= len(generated_strs)
    rougel /= len(generated_strs)
    rougelsum /= len(generated_strs)
    bleu = sacrebleu.corpus_bleu(generated_strs, [gold_strs])
    return {'vloss': vloss,
            'rouge1': vloss.new_zeros(1)+ rouge1,
            'rouge2': vloss.new_zeros(1) + rouge2,
            'rougeL': vloss.new_zeros(1) + rougel,
            'rougeLsum': vloss.new_zeros(1) + rougelsum,
            'bleu': vloss.new_zeros(1)+ bleu.score
           }


def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
