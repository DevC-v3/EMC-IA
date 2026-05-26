import cv2
import mediapipe as mp
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. CARGAR Y PREPARAR EL DATASET
ARCHIVO_CSV = 'dataset.csv'
LETRAS_VALIDAS = ['C', 'A', 'J', 'E']

datos = []
etiquetas = []
with open(ARCHIVO_CSV, mode='r') as f:
    reader = csv.reader(f)
    encabezado = next(reader)
    for fila in reader:
        letra = fila[-1].strip().upper()
        if letra in LETRAS_VALIDAS:
            caracteristicas = list(map(float, fila[:-1]))
            datos.append(caracteristicas)
            etiquetas.append(letra)

X = np.array(datos)
y = np.array(etiquetas)

codificador = LabelEncoder()
y_cod = codificador.fit_transform(y)   # p.ej. A→0, C→1, E→2, J→3

print(f"Clases encontradas: {codificador.classes_}")
print(f"Total de muestras: {len(X)}")

# 2. ENTRENAR EL CLASIFICADOR
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y_cod)
print("Modelo entrenado correctamente.")

# 3. CONFIGURAR MEDIAPIPE Y LA CÁMARA
mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
manos = mp_manos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

print("Iniciando reconocimiento. Presiona ESC para salir.")

# 4. BUCLE DE RECONOCIMIENTO EN TIEMPO REAL
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = manos.process(rgb)

    if resultado.multi_hand_landmarks:
        descriptores = resultado.multi_hand_landmarks[0]

        # Dibujar los landmarks y conexiones
        mp_dibujo.draw_landmarks(
            frame,
            descriptores,
            mp_manos.HAND_CONNECTIONS
        )

        # Extraer coordenadas relativas a la muñeca (landmark 0)
        base_x = descriptores.landmark[0].x
        base_y = descriptores.landmark[0].y
        base_z = descriptores.landmark[0].z

        caracteristicas = []
        for lm in descriptores.landmark:
            caracteristicas.extend([
                lm.x - base_x,
                lm.y - base_y,
                lm.z - base_z
            ])

        # Predecir con el modelo (necesita forma (1, 63))
        pred_cod = modelo.predict([caracteristicas])[0]
        letra_predicha = codificador.inverse_transform([pred_cod])[0]

        # Obtener la probabilidad/confianza (opcional)
        probas = modelo.predict_proba([caracteristicas])[0]
        confianza = np.max(probas) * 100

        # Mostrar la predicción en pantalla
        texto = f"Letra: {letra_predicha} ({confianza:.1f}%)"
        cv2.putText(frame, texto, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Mano no detectada", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Reconocimiento de Letras', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
manos.close()