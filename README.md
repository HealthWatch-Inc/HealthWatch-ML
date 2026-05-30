# 🧠 HealthWatch - Machine Learning Module

Este repositorio contiene los modelos de Inteligencia Artificial encargados del procesamiento de datos biométricos y de movimiento para el ecosistema **HealthWatch**. Nuestro objetivo es transformar los datos crudos de los sensores en información accionable (detección de caídas, nivel de oxígeno y ritmo cardíaco).

## 🚀 Estructura de la Organización
Este proyecto es parte de un sistema integrado:
*   **IoT:** Captura de datos con sensores físicos.
*   **Backend:** Gestión de usuarios y almacenamiento.
*   **App:** Visualización y alertas en tiempo real.
*   **ML (Este repo):** Inteligencia y predicción.

---

## 📂 Estructura del Repositorio

El proyecto se divide en dos módulos principales según el origen de los datos:

### 📦 Módulo A: Vital Signs (SPO2 & Heart Rate)
**Sensor:** `MAX30100`
*   **Objetivo:** Análisis de niveles de saturación de oxígeno en sangre (SpO2) y frecuencia cardíaca.
*   **Modelos:** Algoritmos de filtrado y regresión para limpieza de ruido (ruido por movimiento o mala colocación).
*   **Carpeta:** `/modulo-a-spo2`

### 📦 Módulo B: Activity Recognition & Fall Detection
**Sensor:** `MPU6050` (Acelerómetro y Giroscopio)
*   **Objetivo:** Detección de Caídas Libres e impactos, y Reconocimiento de Actividad Física (HAR).
*   **Algoritmo:** Red Neuronal Recurrente (**LSTM**) implementada en **PyTorch**.
*   **Funcionamiento:** Analiza ventanas temporales de datos (`acc`, `gyro`, `pitch`, `roll`, `yaw`) para identificar el patrón característico de una caída (pérdida de gravedad seguida de impacto y rotación caótica).
*   **Carpeta:** `/modulo-b-acelerometro`

---

## 🛠️ Stack Tecnológico
*   **Lenguaje:** Python 3.11+
*   **Deep Learning:** PyTorch (LSTMs para series temporales)
*   **Procesamiento de Datos:** Pandas, NumPy, Scikit-Learn
*   **Versionamiento de Modelos:** Archivos `.pth` (PyTorch State Dict)

---

## 🔧 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/TuOrganizacion/HealthWatch-ML.git
   cd HealthWatch-ML
   ```

2. **Crear entorno virtual e instalar dependencias:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   pip install torch pandas numpy scikit-learn
   ```

---

## 📲 Integración con la App Móvil
Los modelos entrenados se exportan a la carpeta `/models`. 
*   **Backend/App:** Pueden consumir estos modelos mediante un microservicio en Flask/FastAPI o convirtiéndolos a **TorchScript / ONNX** para ejecución directa en dispositivos móviles (Edge AI).
