import cv2

# Ouvre une vidéo
cap = cv2.VideoCapture("/home/joannes/Vidéos/nuitsdesbassins/output_camera_03.avi")

# Crée un tracker (ici CSRT, assez robuste)
tracker = cv2.legacy.TrackerCSRT_create()

# Lis la première frame
ret, frame = cap.read()
if not ret:
    print("Impossible de lire la vidéo")
    exit()

# Sélectionne une boîte manuellement
bbox = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
print('-------bbox--------', type(bbox),  bbox)

# Initialise le tracker
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mets à jour la position
    success, bbox = tracker.update(frame)

    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Perdu", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Esc pour quitter
        break

cap.release()
cv2.destroyAllWindows()
