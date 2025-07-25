import socketio
import asyncio

# Adresse du serveur Node.js
SERVER_URL = "http://localhost:3001"  # ou adresse Heroku
SERVER_URL = "https://nuit-des-bassins-client-9b7778c21473.herokuapp.com"
# Client Socket.IO asynchrone
sio = socketio.AsyncClient()

# Connexion réussie
@sio.event
async def connect():
    print("✅ Connecté au serveur Node.js")
    # Émettre un événement pour demander les infos utilisateur
    await sio.emit("get_user_data", {"id": "id-joannes"})
    await sio.emit("send_message", {"target": "all", "message": "test coucou 01", "notification": False})
    await sio.emit("update_pseudo", {"id": "id-joannes", "pseudo": "joannes"})


# Déconnexion
@sio.event
async def disconnect():
    print("❌ Déconnecté du serveur Node.js")

# Réception des données utilisateur
@sio.on("user_data")
async def on_user_data(data):
    print("📦 Données utilisateur reçues :", data)

# Réception d’un message
@sio.on("emit_message")
async def on_emit_message(data):
    print("📢 Nouveau message reçu :", data)

# Réception quand le pseudo est mis à jour
@sio.on("pseudo_updated")
async def on_pseudo_updated(data):
    print("📝 Pseudo mis à jour :", data)

# Tu peux écouter d'autres événements comme "continuous_data", "selfie", etc.

# Lancer le client
async def main():
    try:
        await sio.connect(SERVER_URL, transports=["websocket"])
        await sio.wait()  # Attend les événements indéfiniment
    except Exception as e:
        print("❌ Erreur de connexion :", e)

if __name__ == "__main__":
    asyncio.run(main())
