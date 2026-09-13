import torch
import os
import json
import datetime
from utils import write_log
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import transformers
from transformers import AdamW, TrainingArguments
from metrics import preference_bias
import numpy as np


class TrainerForMulticlassClassification:
    def __init__(self,
                 args: TrainingArguments = None,
                 total_steps: int = None,
                 deciding_metric: str = None,
                 class_weights: torch.Tensor = None,
                 id2label: dict = None,
                 model: nn.Module = None,
                 train_loader: DataLoader = None,
                 valid_loader: DataLoader = None,
                 test_loader: DataLoader = None,
                 save_cases: bool = True,
                 keep_only_best_checkpoint: bool = False,
                 restore_train_after_eval: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.total_steps = total_steps
        self.deciding_metric = deciding_metric
        self.model = model
        self.train_loader = train_loader
        self.id2label = id2label
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_cases = save_cases
        self.keep_only_best_checkpoint = keep_only_best_checkpoint
        self.restore_train_after_eval = restore_train_after_eval
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.best_ckpt = None
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        if not os.path.exists(self.args.logging_dir):
            os.makedirs(self.args.logging_dir)

    def evaluate(self, loader, case_study=False):
        cases = []
        self.model.eval()
        with torch.no_grad():
            predictions = []
            truths = []
            bar = tqdm(loader)
            for _, batch in enumerate(bar):
                outputs = self.model(batch)
                y_pred = outputs.get("logits")
                if case_study:
                    contexts = batch["dialogue_history"]
                    preds = torch.argmax(y_pred, dim=-1).int().cpu().detach().numpy()
                    labels = batch.get("label").detach().numpy()
                    attention_weights = []
                    graph_size = len(outputs["graphs"][0]["nodes"]) - 1
                    for w in outputs["attention_weights"]:
                        attention_weights.append(w[1].detach().cpu().numpy()[-graph_size:].tolist())
                    attention_weights = np.array(attention_weights).squeeze().transpose().tolist()
                    for i in range(len(contexts)):
                        cases.append({
                            "dialogue_history": [contexts[i], ],
                            "strategy_history": [batch["strategy_history"][i], ],
                            "speaker_turn": [str(batch["speaker_turn"][i]), ],
                            "prediction": self.id2label[preds[i]],
                            "label": self.id2label[labels[i]],
                            "graph": outputs["graphs"],
                            "attention_weights": attention_weights,
                            "erc_logits": outputs["erc_logits"].cpu().detach().numpy().tolist()
                        })
                predictions.append(torch.argmax(y_pred, dim=-1))
                truths.append(batch.get("label"))
            predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy()
            truths = torch.cat(truths, dim=-1).detach().numpy()
            acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average=None)
            weighted_f1 = f1_score(truths, predictions, average='weighted')
            macro_f1 = f1_score(truths, predictions, average='macro')
            micro_f1 = f1_score(truths, predictions, average='micro')
            c_matrix = confusion_matrix(truths, predictions)
            metrics = {
                'accuracy': acc,
                'macro f1': macro_f1,
                'micro f1': micro_f1,
                'weighted f1': weighted_f1,
                'confusion matrix': c_matrix.tolist(),
                'preference bias': preference_bias(c_matrix)
            }
            for _id in range(len(self.id2label.keys())):
                metrics[self.id2label[_id]] = f1[_id]
            
            # Monitoring structural innovation: Print lambda_prior values
            if (
                hasattr(self.model, 'conv1')
                and hasattr(self.model.conv1, 'conv')
                and hasattr(self.model.conv1.conv, 'lambda_prior')
            ):
                lambdas = self.model.conv1.conv.lambda_prior.detach().cpu().numpy()
                # Print as a formatted list for readability
                lambda_str = ", ".join([f"{l:.4f}" for l in lambdas])
                print(f"\n>>> [Soft-Masking Monitor] lambda_prior: [{lambda_str}]")

            if case_study:
                return metrics, cases
            return metrics

    def train(self):
        log_path = os.path.join(self.args.logging_dir, f'train_{self.timestamp}.log')

        best_checkpoint = 0
        best_metric = 0

        no_decay = ["bias", "LayerNorm.weight"]
        
        # Differential learning rates: give structural innovation a boost
        lambda_params = []
        base_params_decay = []
        base_params_no_decay = []
        
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lambda_prior" in n:
                lambda_params.append(p)
            elif any(nd in n for nd in no_decay):
                base_params_no_decay.append(p)
            else:
                base_params_decay.append(p)

        optimizer_grouped_params = [
            {
                "params": lambda_params,
                "lr": 5e-4,  # High LR for the 19 structural parameters
                "weight_decay": 0.0,
            },
            {
                "params": base_params_decay,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": base_params_no_decay,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_params, lr=self.args.learning_rate)
        total_steps = self.args.num_train_epochs * len(self.train_loader)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=self.args.warmup_steps,
                                                                              num_training_steps=int(total_steps))
        step_counter = 0
        for epoch in range(1, int(self.args.num_train_epochs) + 1):
            stop_training = 0
            self.model.train()
            total_loss = 0
            loss = torch.tensor(0)
            bar = tqdm(self.train_loader)
            for idx, batch in enumerate(bar):
                optimizer.zero_grad()
                step_counter += 1
                bar.set_description(f"Epoch {epoch}| Step {step_counter} | Loss: {loss:.4f}")
                outputs = self.model(batch)
                loss = self.cross_entropy_loss(outputs.get("logits"), batch.get("label").to(self.device))
                loss.backward()
                optimizer.step()
                scheduler.step()
                if step_counter % self.args.save_steps == 0 and not self.keep_only_best_checkpoint:
                    if not os.path.exists(self.args.output_dir):
                        os.makedirs(self.args.output_dir)
                    save_path = os.path.join(self.args.output_dir, f"checkpoint-{step_counter}.pth")
                    self.model.save(save_path)
                if step_counter % self.args.eval_steps == 0:
                    metrics = self.evaluate(self.valid_loader)
                    if metrics[self.deciding_metric] > best_metric:
                        best_metric = metrics[self.deciding_metric]
                        best_checkpoint = step_counter
                        if self.keep_only_best_checkpoint:
                            os.makedirs(self.args.output_dir, exist_ok=True)
                            save_path = os.path.join(self.args.output_dir, "best.pth")
                            self.model.save(save_path)
                    msg = f"Evaluation Step {step_counter} | "
                    for k, v in metrics.items():
                        if k != "confusion matrix":
                            msg += f"{k}: {v:.4f}, "
                    print(msg)
                    write_log(msg, log_path)
                    if self.restore_train_after_eval:
                        self.model.train()
                if step_counter == self.total_steps:
                    stop_training = 1
                    break
            if stop_training:
                break
        print(f"Best checkpoint: Step {best_checkpoint}")
        self.best_ckpt = best_checkpoint
        test_metrics, result_path, checkpoint_path = self.test()
        model_args = getattr(self.model, 'args', None)
        run_summary = {
            "dataset": getattr(model_args, 'dataset', None),
            "expert_fusion": getattr(model_args, 'expert_fusion', None),
            "prior_strength_init": getattr(model_args, 'prior_strength_init', None),
            "seed": getattr(model_args, 'seed', None),
            "best_validation_metric_name": self.deciding_metric,
            "best_validation_metric": float(best_metric),
            "best_checkpoint_step": int(best_checkpoint),
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "result_path": os.path.abspath(result_path),
            "test_metrics": test_metrics,
        }
        summary_path = os.path.join(self.args.logging_dir, f'run_summary_{self.timestamp}.json')
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(run_summary, summary_file, ensure_ascii=False, indent=2)
        print(f"Run summary: {summary_path}")

    def test(self, ckpt=None, checkpoint_path=None):
        print("Testing ...")
        if checkpoint_path:
            ckpt_path = checkpoint_path
            checkpoint_label = checkpoint_path
        else:
            load_ckpt = ckpt if ckpt else self.best_ckpt
            filename = "best.pth" if self.keep_only_best_checkpoint else f"checkpoint-{load_ckpt}.pth"
            ckpt_path = os.path.join(self.args.output_dir, filename)
            checkpoint_label = load_ckpt
        self.model.load(ckpt_path)
        if self.save_cases:
            test_metrics, cases = self.evaluate(self.test_loader, case_study=True)
        else:
            test_metrics = self.evaluate(self.test_loader, case_study=False)
            cases = None
        report_path = os.path.join(self.args.logging_dir, f'result_{self.timestamp}.json')
        json.dump(test_metrics, open(report_path, "w", encoding="utf-8"))
        if self.save_cases:
            cases_path = os.path.join(self.args.logging_dir, f'cases_{self.timestamp}.json')
            json.dump(cases, open(cases_path, "w", encoding="utf-8"))
        msg = f"Test result for checkpoint {checkpoint_label} | "
        for k, v in test_metrics.items():
            if k != "confusion matrix":
                msg += f"{k}: {v:.4f}, "
        print(msg)
        return test_metrics, report_path, ckpt_path
