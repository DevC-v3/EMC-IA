import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from config import ARCHIVO_CSV, LETRAS_VALIDAS, RUTA_MODELO, RUTA_CLASES, NUM_CARACTERISTICAS

# 1. CARGAR Y PREPARAR EL DATASET
datos = []
etiquetas = []

with open(ARCHIVO_CSV, mode='r') as f:
    reader = csv.reader(f)
    next(reader)  # saltar encabezado
    for fila in reader:
        letra = fila[-1].strip().upper()
        if letra in LETRAS_VALIDAS:
            caracteristicas = list(map(float, fila[:-1]))
            datos.append(caracteristicas)
            etiquetas.append(letra)

X = np.array(datos)
y = np.array(etiquetas)

codificador = LabelEncoder()
y_cod = codificador.fit_transform(y)
num_clases = len(codificador.classes_)

print(f"Clases encontradas : {codificador.classes_}")
print(f"Total de muestras  : {len(X)}")

y_onehot = keras.utils.to_categorical(y_cod, num_classes=num_clases)

# División 80 % entrenamiento / 20 % validación
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_cod
)

# 2. CONSTRUIR LA RED NEURONAL
modelo = keras.Sequential([
    layers.Input(shape=(NUM_CARACTERISTICAS,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_clases, activation='softmax')
], name='clasificador_letras')

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

# 3. ENTRENAR
parada_temprana = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

modelo.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[parada_temprana],
    verbose=1
)

loss_val, acc_val = modelo.evaluate(X_val, y_val, verbose=0)
print(f"\nPrecisión en validación : {acc_val * 100:.2f} %")

# 4. GUARDAR MODELO Y CLASES
modelo.save(RUTA_MODELO)
np.save(RUTA_CLASES, codificador.classes_)

print(f"Modelo guardado  → {RUTA_MODELO}")
print(f"Clases guardadas → {RUTA_CLASES}")