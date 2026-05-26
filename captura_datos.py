import cv2
import mediapipe as mp
import os
import csv
from datetime import datetime

LETRA = input('Ingrese la letra a capturar: ').upper()
ARCHIVO_CSV = 'dataset.csv'
CARPETA_IMAGENES = 'capturas'

os.makedirs(CARPETA_IMAGENES, exist_ok=True)
existe_csv = os.path.exists(ARCHIVO_CSV)
contador = 0

# Registro mi dispositivo de captura - 0 indice
cap = cv2.VideoCapture(0)

# Detector de mano
mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

with open(ARCHIVO_CSV, mode='a', newline="") as f:
    writer = csv.writer(f)

    if not existe_csv:
        # Encabezado
        columnas = []
        for i in range(21):
            columnas += [f"x{i}", f"y{i}", f"z{i}"]

        columnas.append("letra")
        writer.writerow(columnas)

    # Configurar el detector
    with mp_manos.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as manos:

        # Bucle infinito
        while True:

            # Capturo Fotograma
            ret, frame = cap.read()

            # Si no puede capturar sale
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_original = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = manos.process(rgb)

            if resultado.multi_hand_landmarks:

                descriptores_mano = resultado.multi_hand_landmarks[0]

                mp_dibujo.draw_landmarks(
                    frame,
                    descriptores_mano,
                    mp_manos.HAND_CONNECTIONS
                )

                cv2.putText(
                    frame,
                    f"Letra: {LETRA} | Presione S para guardar",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Capturas: {contador}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # Muestra el fotograma
            cv2.imshow('Captura de Datos', frame)

            # Sale si presionas ESC
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break

            if tecla == ord("s") and resultado.multi_hand_landmarks:

                contador += 1

                nombre_imagen = (
                    f"{LETRA}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )

                ruta_imagen = os.path.join(
                    CARPETA_IMAGENES,
                    nombre_imagen
                )

                cv2.imwrite(ruta_imagen, frame_original)

                base_x = descriptores_mano.landmark[0].x
                base_y = descriptores_mano.landmark[0].y
                base_z = descriptores_mano.landmark[0].z

                fila = []

                for lm in descriptores_mano.landmark:
                    fila.extend([
                        lm.x - base_x,
                        lm.y - base_y,
                        lm.z - base_z
                    ])

                fila.append(LETRA)

                writer.writerow(fila)

                print(f"Guardado: {LETRA} - {ruta_imagen}")

cap.release()
cv2.destroyAllWindows()