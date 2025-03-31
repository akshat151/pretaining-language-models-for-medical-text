from tqdm import tqdm
from ...env import ProjectPaths
import yaml
from .factory import ModelFactory
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainer.load_config()
        self.model_factory = ModelFactory()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = self.config['training']['epochs']


    @staticmethod
    def load_config():
        "Loads the `config.yaml` file into a structured object"

        config_path = ProjectPaths.CONFIG_DIR / 'config.yaml'
        
        with open(config_path, 'rb') as f:
            return yaml.safe_load(f)
        

    def train(self, trainloader: DataLoader, valloader: DataLoader, dataset: str = 'medal'):
        best_model = None
        best_acc = 0
        
        for model_name in self.config['model_names']:
            print(f'------- {model_name} --------')
            training_hyperparams = self.config['training']['hyperparameters']
            model_hyperparams = self.config['models'][model_name]['hyperparameters']
            model_baseparams = self.config['models'][model_name]['base_params']
            num_classes = self.config['datasets'][dataset]['num_classes']
            
            model_params = {
                **model_hyperparams,
                **model_baseparams,
                **num_classes
            }

            model: nn.Module = self.model_factory.get_model(model_name, model_params).to(self.device)
            loss_fn = self.config['datasets'][dataset]['loss_function']
            if loss_fn == 'cross_entropy':
                self.criterion = CrossEntropyLoss()
            
            optimizer_name = self.config['training']['optimizer']

            if optimizer_name == 'adam':
                self.optimizer = Adam(
                    model.parameters(),
                    lr=training_hyperparams['learning_rate'],
                    weight_decay=self.config['training']['weight_decay'])

            results = {}  

            for epoch in self.num_epochs:
                
                for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), desc='Training'):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
            
                avg_loss = total_loss / len(trainloader)
                accuracy = 100 * correct / total
                
                print(f'Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
                
                results[f'epoch_{epoch+1}'] = {'loss': avg_loss, 'accuracy': accuracy}
            
            val_loss, val_acc = self.evaluate(valloader, model, 'Validation')
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model

            results[f'validation'] = {f'loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}'}
                
        return results
    

    def evaluate(self, dataloader: DataLoader, model: nn.Module, set: str = 'Validation'):
        model.eval()  
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():  
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples

        print(f'{set} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        return avg_loss, accuracy


            

            
    

