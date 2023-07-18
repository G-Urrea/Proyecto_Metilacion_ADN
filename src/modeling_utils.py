'''
Funciones utiles para el modelamiento
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, in_features):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(in_features=in_features, out_features=10)
        self.hidden2 = nn.Linear(in_features=10, out_features=20)
        self.hidden3 = nn.Linear(in_features=20, out_features=30)
        self.hidden4 = nn.Linear(in_features=30, out_features=20)
        self.output = nn.Linear(in_features=20, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout(x)
        x = self.relu(self.hidden4(x))
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            output = np.where(output.detach().numpy()>0.5, 1 , 0)
        return output
    
def simple_binary_train(model, train_dataloader, val_dataloader, epochs=50, report_every=10):
    # Definir función de pérdida y optimizador
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento de la red neuronal
    epochs = epochs
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Modo de entrenamiento
        train_loss = 0.0

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # Evaluación en el conjunto de validación
        model.eval()  # Modo de evaluación
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.unsqueeze(1))
                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_dataloader.dataset)
            val_losses.append(val_loss)
        if epoch%report_every==0:
            # Imprimir métricas de entrenamiento y validación
            print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses

def train_plot(train_losses, val_losses):
    # Gráfico de la pérdida en función de la época
    epochs = len(train_losses)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def create_dataloaders(x_train, y_train, x_val, y_val):
    # Convertir los datos a tensores de PyTorch
    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


    # Crear conjuntos de datos y dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader

def bar_metrics(y_test, y_pred, title="Model Performance Metrics de Random Forest usando data set reducido mediante ANOVA (test set)"):
    # Métricas
    
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    rec = round(recall_score(y_test, y_pred), 2)
    prec= round(precision_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred), 2)
    cohen_kappa = round(cohen_kappa_score(y_test, y_pred), 2)
    roc_auc = round(roc_auc_score(y_test, y_pred), 2)

    # Crear gráfico de barras
    metrics = ["Accuracy", "Recall", "Precision", "F1 Score", "Cohen's Kappa", "ROC AUC"]
    values = [accuracy, rec, prec, f1, cohen_kappa, roc_auc]

    plt.figure()
    plt.bar(metrics, values)
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title(title)

    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()

def metrics_heatmap(y_test, predictions, title=''):
    metrics_dict = {'Metrics':['Accuracy', 'Accuracy(Balanced)', 'Recall', 'Precision', 'F1', 'Cohen Kappa', 'ROC AUC']}
    for estim in predictions:
        
        y_pred = predictions[estim]
        accuracy = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec= precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        metrics_dict[estim] = [accuracy, bal_acc, rec, prec, f1, cohen_kappa, roc_auc]
    sns.heatmap(pd.DataFrame(metrics_dict).set_index('Metrics'), annot=True, fmt=".3f")
    plt.xticks(rotation=45)
    plt.title(title)

def fit_estimators_from_dict(x,  y, basic_estimators):
    '''
        - basic_estimators: diccionario con tipo de estimadores {'clf': clf}
    '''
    fitted_dict = {}
    for estimator in basic_estimators:
        model = clone(basic_estimators[estimator])
        name = type(model).__name__
        fitted_dict[name] = model
        fitted_dict[name].fit(x, y)
        
    return fitted_dict

def predict_from_dict(x, estimators):
    '''
        - estimators: diccionario con estimadores fitteados
    '''
    preds = {}
    for estim in estimators:
        preds[estim] = estimators[estim].predict(x)
    return preds