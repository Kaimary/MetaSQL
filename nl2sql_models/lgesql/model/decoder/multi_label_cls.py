import torch
import torch.nn as nn
from model.model_utils import Registrable

@Registrable.register('multi_label_cls')
class MultiLabelClassifier(nn.Module):
    def __init__(self, args):
        super(MultiLabelClassifier, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(args.lstm_hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(256, 25),
            nn.Linear(25, 25),
            nn.Dropout(p=0.0),
        )
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, batch):
        logits = self.feedforward(inputs)
        # probabilities = torch.nn.functional.sigmoid(logits)
        labels = batch.labels.clone().detach().float().to(logits.device)
        # labels = torch.tensor(batch.labels, dtype=torch.float32, device=logits.device)
        loss = self.loss_func(logits + 1e-8, labels)
        
        return loss
    
    def inference(self, inputs, batch):
        logits = self.feedforward(inputs)
        labels = batch.labels.clone().detach().float().to(logits.device)
        predictions = (logits.data > 0.0).long()
        label_data = labels.data.long()
        
        true_positives = (predictions * label_data).sum().item()
        false_positives = (predictions * (1 - label_data)).sum().item()
        true_negatives = ((1-predictions)* ( 1- label_data)).sum().item()
        false_negatives = ((1-predictions) * label_data).sum().item()

        return true_positives, false_positives, true_negatives, false_negatives
    

    def test(self, inputs):
        logits = self.feedforward(inputs)
        
        return logits.cpu().data.numpy()