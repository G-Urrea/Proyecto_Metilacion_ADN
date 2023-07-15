'''
Funciones utiles para el modelamiento
'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, roc_auc_score
import matplotlib.pyplot as plt

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
    
def simple_binary_train(model, train_dataloader, val_dataloader, epochs=50):
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

        # Imprimir métricas de entrenamiento y validación en cada época
        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses


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