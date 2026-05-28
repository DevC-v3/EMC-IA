import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from config import RUTA_MODELO, RUTA_CLASES

# 1. CARGAR MODELO Y CLASES GUARDADAS
modelo = keras.models.load_model(RUTA_MODELO)
clases = np.load(RUTA_CLASES, allow_pickle=True)

codificador = LabelEncoder()
codificador.classes_ = clases

print(f"Modelo cargado   : {RUTA_MODELO}")
print(f"Clases cargadas  : {clases}")

# 2. CONFIGURAR MEDIAPIPE Y LA CÁMARA
mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

manos = mp_manos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

print("Iniciando reconocimiento en tiempo real. Presiona ESC para salir.")

# 3. BUCLE DE RECONOCIMIENTO EN TIEMPO REAL
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = manos.process(rgb)

    if resultado.multi_hand_landmarks:
        descriptores = resultado.multi_hand_landmarks[0]

        # Dibujar landmarks y conexiones
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

        # Predicción con la red neuronal
        entrada = np.array([caracteristicas], dtype=np.float32)
        probabilidades = modelo.predict(entrada, verbose=0)[0]

        indice_pred = np.argmax(probabilidades)
        letra_predicha = codificador.inverse_transform([indice_pred])[0]
        confianza = probabilidades[indice_pred] * 100

        # Mostrar resultado en pantalla
        texto = f"Letra: {letra_predicha} ({confianza:.1f}%)"
        cv2.putText(frame, texto, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Mano no detectada", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Reconocimiento de Letras - Red Neuronal', frame)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
manos.close()