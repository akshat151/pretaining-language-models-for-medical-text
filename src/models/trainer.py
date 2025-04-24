from tqdm import tqdm
from env import ProjectPaths
import yaml
from .factory import ModelFactory
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json




class ModelTrainer:
    def __init__(self, config_file: str, pretrained_model_path: str = None):
        self.config = ModelTrainer.load_config(config_file)
        self.model_factory = ModelFactory()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = self.config['training']['epochs']
        trained_models_dir = ProjectPaths.PROJECT_DIR.value / 'trained_models' / 'models'
        trained_models_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_model = None
        if pretrained_model_path:
            print('Using pretrained model for training')
            self.pretrained_model = torch.load(
                pretrained_model_path, 
                map_location=self.device
            )
        self.dataset_name = None


    @staticmethod
    def load_config(config_file: str):
        "Loads the `config.yaml` file into a structured object"

        config_path = ProjectPaths.CONFIG_DIR.value / config_file
        
        with open(config_path, 'rb') as f:
            return yaml.safe_load(f)
        
        
    def save_model(self, model, model_name, save_dir='trained_models/models/'):
        """Save the model state dictionary to a file"""
        save_path = f"{save_dir}{model_name}_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")


    def init_weights(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Apply Xavier initialization for LSTM layers
                    nn.init.xavier_uniform_(param.data)
                else:
                    # Apply He initialization for other layers (if applicable)
                    nn.init.kaiming_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


    def train(self, trainloader, valloader, dataset: str, embedding_dim=100, **kwargs):
        self.dataset_name = dataset
        for model_name in self.config['model_names']:
            print(f'------- {model_name} --------')

            training_hyperparams = self.config['training']['hyperparameters']
            model_hyperparams = self.config['models'][model_name].get('hyperparameters', {})
            model_baseparams = self.config['models'][model_name].get('base_params', {})
            num_classes = self.config['datasets'][dataset]['num_classes']
            create_embedding_layer = self.config['training']['create_embedding_layer']
            embedding_model = kwargs.get('embedding_model', None)

            model_params = {
                **(model_hyperparams if model_hyperparams else {}),
                **(model_baseparams if model_baseparams else {}),
                'num_classes': num_classes,
                'embedding_dim': embedding_dim,
                'create_embedding_layer': create_embedding_layer,
                'embedding_model': embedding_model
            }

            print(model_params)
            model: nn.Module = self.model_factory.get_model(model_name, **model_params).to(self.device)
            # Attach hook
            # hook_handle = model.fc_embedding.register_forward_hook(self.monitor_fc_embedding_output)
            # model.fc_embedding.weight.register_hook(self.monitor_fc_embedding_grad)
            
            if self.pretrained_model:
                model_state = model.state_dict()
                filtered_state = {k: v for k, v in self.pretrained_model.items() if k in model_state and v.size() == model_state[k].size()}
                model_state.update(filtered_state)
                model.load_state_dict(model_state)
                print(f"Loaded {len(filtered_state)} parameters from pretrained model.")
                model.fc1 = nn.Linear(4 * model_hyperparams['lstm_hidden_dim'], num_classes).to(self.device)
                torch.nn.init.xavier_uniform_(model.fc1.weight)


            loss_fn = self.config['datasets'][dataset]['loss_function']
            label_smoothing = self.config.get('label_smoothing', 0.1)
            class_weights = None

            if 'class_weights' in self.config['datasets'][dataset]:
                with open(self.config['datasets'][dataset]['class_weights'], 'r') as f:
                    weights = json.load(f)

                # Convert to torch tensor and move to device
                class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            

            if self.dataset_name == 'mimic-iv':
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)

            optimizer_name = self.config['training']['optimizer']
            learning_rate = training_hyperparams['learning_rate']
            weight_decay = self.config['training']['weight_decay']

            if optimizer_name == 'adam':
                self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            grad_accum_steps = training_hyperparams.get('gradient_accumulation_steps', 1)
            total_training_steps = (len(trainloader) // grad_accum_steps) * self.num_epochs
            warmup_steps = training_hyperparams.get('warmup_steps', 500)

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / max(1, warmup_steps)
                progress = float(current_step - warmup_steps) / max(1, total_training_steps - warmup_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = LambdaLR(self.optimizer, lr_lambda)

            results = {}
            best_model = None
            best_acc = 0

            for epoch in range(self.num_epochs):
                total_loss = 0.0
                running_loss = 0.0
                all_targets = []
                all_preds = []

                model.train()

                if epoch + 1 == 5:
                    model.unfreeze_embeddings()

                self.optimizer.zero_grad()

                for batch_idx, (inputs, masks, targets) in tqdm(enumerate(trainloader), desc='Training', total=len(trainloader)):
                    inputs, masks = inputs.to(self.device), masks.to(self.device)
                    outputs = model(inputs, masks)

                    if self.dataset_name == 'mimic-iv':
                        targets = targets.to(self.device)
                        loss = self.criterion(outputs, targets.float())  # BCEWithLogitsLoss expects target to be float
                    else:
                        targets = targets.to(self.device)
                        loss = self.criterion(outputs, targets)  # For multi-class, targets should be indices

                    loss = loss / grad_accum_steps
                    loss.backward()
                    # self.log_gradients(model)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    total_loss += loss.item() * grad_accum_steps
                    running_loss += loss.item() * grad_accum_steps

                    if self.dataset_name == 'mimic-iv':
                        # Apply sigmoid activation to outputs to get probabilities
                        probs = torch.sigmoid(outputs)
                        # Predict labels by thresholding the probabilities at 0.5
                        preds = (probs >= 0.5).float()  # Convert to 1 or 0 based on threshold
                        # Store the targets and predictions for later evaluation
                        all_targets.append(targets.cpu())
                        all_preds.append(preds.cpu())
                    else:
                        preds = outputs.argmax(dim=1)
                        all_targets.extend(targets.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())

                    if (batch_idx + 1) % grad_accum_steps == 0:
                        self.optimizer.step()
                        scheduler.step()
                        self.optimizer.zero_grad()

                    if (batch_idx + 1) % 100 == 0:
                        avg_running_loss = running_loss / 100
                        print(f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(trainloader)} - Avg Loss (last 10000): {avg_running_loss:.4f}")
                        running_loss = 0.0

                avg_loss = total_loss / len(trainloader.dataset)

                if self.dataset_name == 'mimic-iv':
                    # Concatenate all predictions and targets into tensors
                    preds_tensor = torch.cat(all_preds)
                    targets_tensor = torch.cat(all_targets)
                    
                    # Calculate precision@k and recall@k
                    precision_at_k = (preds_tensor * targets_tensor).sum(dim=1) / preds_tensor.sum(dim=1).clamp(min=1)
                    recall_at_k = (preds_tensor * targets_tensor).sum(dim=1) / targets_tensor.sum(dim=1).clamp(min=1)
                    
                    # Compute metrics
                    train_precision = precision_at_k.mean().item()
                    train_recall = recall_at_k.mean().item()
                    train_acc = (preds_tensor == targets_tensor).float().mean().item()

                    print(f"[Epoch {epoch+1}] Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train Accuracy: {train_acc:.4f}")
                
                else:
                    # For multi-class classification (non-MIMIC-IV dataset)
                    train_acc = accuracy_score(all_targets, all_preds)
                    train_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
                    train_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)

                    print(f"[Epoch {epoch+1}] Train Accuracy: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
                
                
                results[f'epoch_{epoch+1}'] = {
                    'loss': avg_loss,
                    'accuracy': train_acc,
                    'precision': train_precision,
                    'recall': train_recall
                }

                # Evaluate on the validation set
                val_loss, val_acc, val_precision, val_recall = self.evaluate(valloader, model, dataset)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model

                results[f'validation_{epoch+1}'] = {
                    'loss': val_loss,
                    'accuracy': val_acc,
                    'precision': val_precision,
                    'recall': val_recall,
                }

                print(f'Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Recall: {train_recall:.4f}, Val Acc: {val_acc:.4f}, Val Prec: {val_precision:.4f}, Val Recall: {val_recall:.4f}')

        self.save_model(best_model, f'mimic_from_scratch_glove_100ctx_{model_name}_model')

        return results


    def monitor_fc_embedding_output(self, module, input, output):
        with torch.no_grad():
            print(f"[fc_embedding] Output stats - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}, Max: {output.max().item():.4f}, Min: {output.min().item():.4f}, NaNs: {torch.isnan(output).sum().item()}")


    def log_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                print(f"{name} grad norm: {grad_norm:.6f}")

    def monitor_fc_embedding_grad(self, grad):
        print(f"[fc_embedding.weight] Grad stats - Mean: {grad.abs().mean().item():.6f}, Max: {grad.abs().max().item():.6f}, NaNs: {torch.isnan(grad).sum().item()}")



    def evaluate(self, dataloader: DataLoader, model: nn.Module, criterion=None, dataset_name: str = 'Validation'):
        model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        
        if not criterion:
            criterion = self.criterion

        with torch.no_grad():
            for inputs, masks, targets in tqdm(dataloader, 'validating', len(dataloader)):
                inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)

                outputs = model(inputs, masks)

                if self.dataset_name == 'mimic-iv':
                    outputs = torch.sigmoid(outputs)

                    # Convert targets to multi-hot (one-hot) vectors
                    multi_hot_targets = torch.zeros_like(outputs).to(targets.device)
                    multi_hot_targets.scatter_(1, targets.unsqueeze(1), 1)

                    # Compute loss
                    loss = criterion(outputs, multi_hot_targets.float())

                    # Top-k = 5 predictions
                    topk_preds = torch.topk(outputs, k=5, dim=1).indices
                    batch_preds = torch.zeros_like(outputs)
                    batch_preds.scatter_(1, topk_preds, 1)

                    all_predictions.append(batch_preds.cpu())
                    all_targets.append(multi_hot_targets.cpu())
                else:
                    loss = criterion(outputs, targets)
                    predictions = outputs.argmax(dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if self.dataset_name == 'mimic-iv':
            preds_tensor = torch.cat(all_predictions)
            targets_tensor = torch.cat(all_targets)
            precision_at_k = (preds_tensor * targets_tensor).sum(dim=1) / preds_tensor.sum(dim=1).clamp(min=1)
            recall_at_k = (preds_tensor * targets_tensor).sum(dim=1) / targets_tensor.sum(dim=1).clamp(min=1)
            precision = precision_at_k.mean().item()
            recall = recall_at_k.mean().item()
            accuracy = (preds_tensor == targets_tensor).float().mean().item()
        else:
            accuracy = accuracy_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)

        print(f'{dataset_name} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision@k: {precision:.4f}, Recall@k: {recall:.4f}')
        return avg_loss, accuracy, precision, recall




    def plot_results(self, results):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        val_precisions = []
        val_recalls = []

        for epoch, metrics in results.items():
            if 'epoch' in epoch:
                train_losses.append(metrics['loss'])
                train_accuracies.append(metrics['accuracy'])
            elif 'validation' in epoch:
                val_losses.append(metrics['loss'])
                val_accuracies.append(metrics['accuracy'])
                val_precisions.append(metrics['precision'])
                val_recalls.append(metrics['recall'])

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_accuracies, label='Train Acc', color='blue')
        plt.plot(val_accuracies, label='Val Acc', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(val_precisions, label='Precision@k', linestyle='--', color='green')
        plt.plot(val_recalls, label='Recall@k', linestyle='-.', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Top-k Precision / Recall (Validation)')
        plt.legend()

        plt.tight_layout()
        plt.show()
