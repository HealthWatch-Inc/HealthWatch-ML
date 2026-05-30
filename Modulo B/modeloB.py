import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURACIÓN Y RUTAS ---
DATA_PATH = "data/"
MODELS_DIR = "models"
WINDOW_SIZE = 20
STEP_SIZE = 5
FEATURES = ['pitch', 'roll', 'yaw', 'gyrox', 'gyroy', 'gyroz', 'accz', 'accx', 'accy']
TARGET = 'secayo'

# Crear carpeta models si no existe
os.makedirs(MODELS_DIR, exist_ok=True)


# --- 2. PREPROCESAMIENTO ---
def prepare_data():
    all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    data_list = [pd.read_csv(f).dropna() for f in all_files]
    full_df = pd.concat(data_list, ignore_index=True)

    # Escalamiento
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_df[FEATURES])

    sequences = []
    labels = []

    for i in range(0, len(scaled_data) - WINDOW_SIZE, STEP_SIZE):
        window = scaled_data[i: i + WINDOW_SIZE]
        # Etiqueta 1 si hubo caída en cualquier momento de la ventana
        label = 1.0 if full_df[TARGET].iloc[i: i + WINDOW_SIZE].any() else 0.0

        sequences.append(window)
        labels.append(label)

    return np.array(sequences), np.array(labels)


# --- 3. DEFINICIÓN DE LA RED NEURONAL (LSTM) ---
class FallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Capa lineal para clasificación binaria
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, time_steps, features)
        out, _ = self.lstm(x)
        # Tomamos solo el último valor de la secuencia (last time step)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# --- 4. ENTRENAMIENTO ---
print("Preparando datos...")
X, y = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a Tensores de PyTorch
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# Instanciar modelo
model = FallLSTM(input_size=len(FEATURES), hidden_size=64, num_layers=2)
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Iniciando entrenamiento...")
epochs = 20
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# --- 5. GUARDAR EL MODELO ---
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "fall_model_pytorch.pth")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModelo guardado exitosamente en: {MODEL_SAVE_PATH}")

# --- 6. EVALUACIÓN RÁPIDA ---
model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_test_t).float().mean()
    print(f"Exactitud en test: {accuracy:.4f}")