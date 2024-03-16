from bart import *
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate using BART')
    parser.add_argument("--default_dir", type=str, default="./",
                        help="default_dir", dest="default_dir")
    
    parser.add_argument("--train_path", type=str, help="path for training dataset json file",
                        default="data/DBLP-QuAD/processed_train_new.json",
                        dest="train_path")
    parser.add_argument("--test_path", type=str, help="path for test dataset json file",
                        default="data/DBLP-QuAD/processed_test_new.json",
                        dest="test_path")
    parser.add_argument("--val_path", type=str, help="path for validation dataset json file",
                        default="data/DBLP-QuAD/processed_valid_new.json",
                        dest="val_path")
    
    parser.add_argument("--save_dir", type=str, default='logs',
                        help="directory to save models.", dest="save_dir")

    parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model",
                        dest="save_prefix")
    
    parser.add_argument("--source", type=str, help="input column, for example: question,aug1,aug2",
                        default="question", dest="source")
    parser.add_argument("--target", type=str,
                        help="query column as ground truth, for example: 'processed_query','processed_query_converted'",
                        default="processed_query", dest="target")
    parser.add_argument("--small_dataset", help="if use small split of datasets to test the scripts",
                        action="store_true",dest="small_dataset")
    parser.add_argument("--save_best", help="save best checkpoint", action="store_true",dest="save_best")
    parser.add_argument("--seed", type=int, default=666, help="seed", dest="seed")
    parser.add_argument("--bart_version", type=str, help="bart-base or bart-large", default="bart-base",
                        dest="bart_version")
    parser.add_argument("--logger_batch_interval", type=int,
                        help="how often will logger show details of training loss and evaluation metrics, unit: batches",
                        default=100, dest="logger_batch_interval")
    parser.add_argument("--progress_bar_refresh_interval", type=int,
                        help="how often progress bar will be refreshed, unit:seconds", default=120,
                        dest="progress_bar_refresh_interval")
    parser.add_argument("--max_length", type=int, help="max_length when encoding", default=512, dest="max_length")
    parser.add_argument("--max_output_length", type=int, default=1024,
                        help="max_output_length for generation", dest="max_output_length")
    parser.add_argument("--min_output_length", type=int, default=8,
                        help="min_output_length for generation", dest="min_output_length")

    parser.add_argument("--batch_size", type=int, help="batch size", default=8, dest="batch_size")
    parser.add_argument("--max_epochs", type=int, default=1000, help="maximum number of epochs", dest="max_epochs")
    parser.add_argument("--patience", type=int, default=30, help="patience for early stopping", dest="patience")
    parser.add_argument("--gpus", type=str, help="IDs of gpus, separated by comma", default="0", dest="gpus")
    parser.add_argument("--device", type=int, help="when using only one gpu, can set device here", default=0,
                        dest="device")

    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", dest="resume_ckpt")
    parser.add_argument("--resume_dir", type=str, default='logs', help="directory to load model from checkpoint",
                        dest="resume_dir")
    parser.add_argument("--resume_prefix", type=str, dest="resume_prefix")
    parser.add_argument("--resume_option", type=int, default=-1,
                        help="-1 means no resuming, 1 means best checkpoint, 0 means last checkpoint",
                        dest="resume_option")
    parser.add_argument("--optim_resume", action="store_true", dest="optim_resume")
    parser.add_argument("--lr_resume", action="store_true", dest="lr_resume")

    parser.add_argument("--learning_rate", help="learning rate", default=1e-5, dest="learning_rate")
    parser.add_argument("--early_stopping_metric", type=str, default='bleu',
                        help="metric to be used for early stopping: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu",
                        dest="early_stopping_metric")
    parser.add_argument("--lr_reduce_patience", type=int, default=4,
                        help="patience for LR reduction in Plateau scheduler", dest="lr_reduce_patience")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5,
                        help="learning rate reduce factor for Plateau scheduler", dest="lr_reduce_factor")
    parser.add_argument("--min_lr", default=1e-8,
                        help="min_lr for Plateau scheduler", dest="min_lr")
    parser.add_argument("--cooldown", type=int, default=1,
                        help="cooldown for Plateau scheduler", dest="cooldown")

    parser.add_argument("--freeze_option", type=int,
                        help="freeze option, positive value means freezing, negative means no freezing",
                        default=-1,
                        dest="freeze_option")
    parser.add_argument("--freeze_embeds", action="store_true",dest="freeze_embeds")
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps",
                        dest="grad_accum")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout",
                        dest="attention_dropout")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout", dest="dropout")
    parser.add_argument("--activation_dropout", type=float, default=0.0, help="activation_dropout",
                        dest="activation_dropout")
    parser.add_argument("--eval_beams", type=int, default=5, help="beam size for inference when testing/validating",
                        dest="eval_beams")
    parser.add_argument("--label_smoothing", type=float, default=0.0, dest="label_smoothing")
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='gradient checkpointing to save memory', dest="gradient_checkpointing")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=50,
                        help="no_repeat_ngram_size when generating predictions",
                        dest="no_repeat_ngram_size")
    args = parser.parse_args()
    print("input args: ", args)

    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    args.learning_rate = float(args.learning_rate)

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main_model = MyModel(args)
    main_model.train()
