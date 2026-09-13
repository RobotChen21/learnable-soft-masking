import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloaders import ESConvPreProcessed, AnnoMIPreProcessed
from transformers import TrainingArguments
from modules.trainer import TrainerForMulticlassClassification
from modules.roberta import RobertaHeterogeneousGraph
from utils import seed_everything

MODELS = {
    "roberta-hg": {
        "model": RobertaHeterogeneousGraph,
        "trainer": TrainerForMulticlassClassification
    }
}

DATASETS = {
    "esconv-preprocessed": ESConvPreProcessed,
    "annomi-preprocessed": AnnoMIPreProcessed,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train', 'test'), default='train')
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--total_epochs', type=int, default=10)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--exclude_others', type=int, default=0)
    parser.add_argument('--erc_temperature', type=float, default=0.5)
    parser.add_argument('--erc_mixed', type=int, default=1)
    parser.add_argument('--hg_dim', type=int, default=512)
    parser.add_argument(
        '--expert_fusion',
        type=str,
        choices=('original_hard', 'learnable_soft'),
        default='learnable_soft',
        help='How discourse-parser structure is fused into the graph.',
    )
    parser.add_argument('--prior_strength_init', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--logging_dir', type=str, default=None)
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Load this exact checkpoint file in test mode.',
    )
    parser.add_argument('--save_cases', type=int, choices=(0, 1), default=1)
    parser.add_argument('--keep_only_best_checkpoint', type=int, choices=(0, 1), default=0)
    parser.add_argument('--restore_train_after_eval', type=int, choices=(0, 1), default=0)

    args = parser.parse_args()
    if args.mode == 'test' and not args.checkpoint_path:
        parser.error('--checkpoint_path is required in test mode')

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODELS[args.model]["model"](args)
    model.to(device)
    train_set = DATASETS[args.dataset]("train", args)
    valid_set = DATASETS[args.dataset]("valid", args)
    test_set = DATASETS[args.dataset]("test", args)
    print(f"Total samples: {len(train_set) + len(valid_set) + len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=valid_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=test_set.collate_fn)

    run_id = f"{args.model}-{args.dataset}-{args.expert_fusion}-seed-{args.seed}"
    output_dir = args.output_dir or os.path.join("runs", run_id, "checkpoints")
    logging_dir = args.logging_dir or os.path.join("runs", run_id, "logs")
    print(
        f"Experiment: fusion={args.expert_fusion}, seed={args.seed}, "
        f"output_dir={output_dir}"
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.total_epochs,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        logging_dir=logging_dir,
        learning_rate=args.lr,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    trainer = MODELS[args.model]["trainer"](
        class_weights=torch.tensor(train_set.class_weights),
        model=model,
        deciding_metric="macro f1",
        args=training_args,
        total_steps=args.total_steps,
        id2label=train_set.id2label,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        save_cases=args.save_cases == 1,
        keep_only_best_checkpoint=args.keep_only_best_checkpoint == 1,
        restore_train_after_eval=args.restore_train_after_eval == 1,
    )

    if args.mode == "train":
        trainer.train()
    else:
        trainer.test(checkpoint_path=args.checkpoint_path)
