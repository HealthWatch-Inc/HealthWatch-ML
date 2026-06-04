import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('models', exist_ok=True)

# -------------------------------
# 1. CONFIGURACIÓN
# -------------------------------
WINDOW_SIZE = 30          # 30 lecturas por ventana
BATCH_SIZE = 32
EPOCHS = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001

# -------------------------------
# 2. CARGAR Y ALINEAR DATOS 
# -------------------------------
df_hr = pd.read_csv('data/heartrate.csv')
df_spo2 = pd.read_csv('data/spo2.csv')
df_temp = pd.read_csv('data/skin_temperature.csv')

print(f"Filas originales - HR: {len(df_hr)}, SpO2: {len(df_spo2)}, Temp: {len(df_temp)}")

min_filas = min(len(df_hr), len(df_spo2), len(df_temp))
if len(df_hr) != min_filas or len(df_spo2) != min_filas or len(df_temp) != min_filas:
    print(f"[ADVERTENCIA] Recortando a {min_filas} filas.")
    df_hr = df_hr.iloc[:min_filas]
    df_spo2 = df_spo2.iloc[:min_filas]
    df_temp = df_temp.iloc[:min_filas]

val_cols = [f'val_{i}' for i in range(WINDOW_SIZE)]
X_hr = df_hr[val_cols].values      # (4265, 30)
X_spo2 = df_spo2[val_cols].values  # (4265, 30)
X_temp = df_temp[val_cols].values   # (4265, 30)
y = df_hr['label'].values

# Mapeo de etiquetas a números
label_mapping = {'okay': 0, 'warning': 1, 'bad': 2}
y_encoded = np.array([label_mapping[label] for label in y])

# -------------------------------
# 3. CREAR SECUENCIAS (ya las tenemos: cada fila es una secuencia de 30 pasos)
# -------------------------------
# Apilamos los tres sensores como features: (muestras, pasos=30, canales=3)
X = np.stack([X_hr, X_spo2, X_temp], axis=2)   # forma final: (4265, 30, 3)

# Normalizar cada canal por separado (importante para LSTM)
scaler = StandardScaler()
# Ajustamos el scaler sobre todas las muestras y pasos, pero por canal
# Para simplificar, redimensionamos a (muestras*pasos, canales)
original_shape = X.shape
X_reshaped = X.reshape(-1, 3)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(original_shape)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convertir a tensores de PyTorch
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)   # clasificación multiclase
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Entrenamiento: {len(X_train)} muestras, Prueba: {len(X_test)} muestras")

# -------------------------------
# 4. DEFINIR EL MODELO LSTM (para clasificación multiclase)
# -------------------------------
class HealthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HealthLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)                # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]                  # Tomamos la última salida de la secuencia
        out = self.fc(out)                   # (batch, num_classes)
        return out

# Instanciar modelo
model = HealthLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=3)
criterion = nn.CrossEntropyLoss()           # Para clasificación multiclase
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# 5. ENTRENAMIENTO
# -------------------------------
print("Iniciando entrenamiento LSTM...")
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)            # (batch, 3)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# -------------------------------
# 6. EVALUACIÓN EN TEST
# -------------------------------
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Reporte de clasificación
print("\n--- Reporte de clasificación (LSTM) ---")
print(classification_report(all_labels, all_preds, 
                            target_names=['okay', 'warning', 'bad']))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['okay','warning','bad'],
            yticklabels=['okay','warning','bad'])
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - LSTM')
plt.show()

# -------------------------------
# 7. GUARDAR MODELO Y ARTEFACTOS
# -------------------------------
# Guardamos el state_dict del modelo y el scaler
torch.save(model.state_dict(), 'models/modelo_lstm_health.pth')
joblib.dump(scaler, 'models/scaler_lstm.pkl')
joblib.dump(label_mapping, 'models/label_mapping_lstm.pkl')

config = {
    'window_size': WINDOW_SIZE,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'input_size': 3,
    'num_classes': 3
}
joblib.dump(config, 'models/config_lstm.pkl')
print("\nModelo LSTM y artefactos guardados en la carpeta 'models'.")

# -------------------------------
# 8. FUNCIÓN DE PREDICCIÓN PARA NUEVOS DATOS (simulando InfluxDB)
# -------------------------------
def predict_lstm(hr_seq, spo2_seq, temp_seq, model, scaler, config):
    """
    hr_seq, spo2_seq, temp_seq: listas de 30 valores cada una (floats)
    Retorna: (categoría, probabilidades)
    """
    # Crear array de forma (1, 30, 3)
    seq = np.stack([hr_seq, spo2_seq, temp_seq], axis=1)   # (30,3) -> luego reshape
    # Normalizar con el mismo scaler
    seq_reshaped = seq.reshape(-1, 3)
    seq_scaled = scaler.transform(seq_reshaped)
    seq_scaled = seq_scaled.reshape(1, 30, 3)
    
    # Convertir a tensor
    seq_tensor = torch.FloatTensor(seq_scaled)
    
    model.eval()
    with torch.no_grad():
        output = model(seq_tensor)          # (1,3)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    
    inv_map = {0:'okay', 1:'warning', 2:'bad'}
    return inv_map[pred_class], probs

# Prueba con la primera muestra del test
sample_idx = 0
hr_s = X_test[sample_idx, :, 0].tolist()
spo2_s = X_test[sample_idx, :, 1].tolist()
temp_s = X_test[sample_idx, :, 2].tolist()
real_label = list(label_mapping.keys())[list(label_mapping.values()).index(y_test[sample_idx])]
pred_label, probs = predict_lstm(hr_s, spo2_s, temp_s, model, scaler, config)
print(f"\n--- Prueba con muestra de test ---")
print(f"Etiqueta real: {real_label}")
print(f"Predicción LSTM: {pred_label} (prob: okay={probs[0]:.3f}, warning={probs[1]:.3f}, bad={probs[2]:.3f})")