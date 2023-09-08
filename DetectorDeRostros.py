import face_recognition
import cv2
import os
import numpy as np

# Cargar las imágenes de referencia desde la carpeta de rostros
folder_path = "C:/Users/marco/Desktop/Deteccion-de-rostros-Clase-8-main/rostros"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image = face_recognition.load_image_file(os.path.join(folder_path, filename))
        face_encoding = face_recognition.face_encodings(image)[0]  # Solo tomamos la primera cara si hay múltiples

        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar caras en el marco de la cámara
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Para cada cara detectada, comparar con las caras conocidas
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparar con las caras conocidas
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        min_distance_index = np.argmin(distances)

        if distances[min_distance_index] < 0.9:
            name = known_face_names[min_distance_index]
        else:
            name = "Desconocido"

        # Dibujamos un rectangulo y ponemos el nombre de la persona
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Color verde: (0, 255, 0)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Color rojo: (0, 0, 255)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
