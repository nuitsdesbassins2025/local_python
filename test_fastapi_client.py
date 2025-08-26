import httpx
import json

FASTAPI_URL = "http://localhost:8000/camera/detection"

def send_camera_detection():
    # Exemple de JSON brut (libre à toi de modifier)
    payload = {
        "tracking_datas": [
            {"tracking_id": "42", "related_client_id": "player1", "posX": 0.5, "posY": 0.8, "zone": "A"},
            {"tracking_id": "43", "related_client_id": "player2", "posX": 0.2, "posY": 0.4, "zone": "B"}
        ],
        "tracking_fps": 30.0
    }

    try:
        response = httpx.post(FASTAPI_URL, json=payload, timeout=10.0)
        print("✅ Réponse du serveur :", response.json())
    except Exception as e:
        print("❌ Erreur lors de l'envoi :", e)


if __name__ == "__main__":
    send_camera_detection()
