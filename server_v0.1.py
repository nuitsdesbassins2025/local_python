import socketio
import asyncio
from aiohttp import web  # pour créer un petit serveur HTTP/Socket.IO


# Adresse du serveur Node.js
REMOTE_SERVER_URL = "http://localhost:3001"  # ou adresse Heroku
REMOTE_SERVER_URL = "https://dopamine.lesateliersdelahalle.fr/"
REMOTE_SERVER_URL = "https://nuit-des-bassins-client-9b7778c21473.herokuapp.com/"
# Client Socket.IO asynchrone
sio_remote = socketio.AsyncClient()



# --- Serveur local WebSocket ---
sio_local = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
app = web.Application()
sio_local.attach(app)



# Connexion réussie
@sio_remote.event
async def connect():
    print("✅ Connecté au serveur Node.js")
    # Émettre un événement pour demander les infos utilisateur
    await sio_remote.emit("get_user_data", {"id": "id-admin1234"})
    await sio_remote.emit("send_message", {"target": "all", "message": "admin connecté !", "notification": False})
    await sio_remote.emit("update_pseudo", {"id": "id-joannes", "pseudo": "id-admin1234"})


# Déconnexion
@sio_remote.event
async def disconnect():
    print("❌ Déconnecté du serveur Node.js")

# Réception des données utilisateur
@sio_remote.on("user_data")
async def on_user_data(data):
    print("📦 Données utilisateur reçues :", data)

# Réception d’un message
@sio_remote.on("emit_message")
async def on_emit_message(data):
    print("📢 Nouveau message reçu :", data)

# Réception quand le pseudo est mis à jour
@sio_remote.on("pseudo_updated")
async def on_pseudo_updated(data):
    print("📝 Pseudo mis à jour :", data)


# Réception continuous data
@sio_remote.on("continuous_data")
async def on_continuous_data(data):
    print("📝 Data mises à jour :", data)


# Réception actions déclenchées
@sio_remote.on("action_triggered_by")
async def on_action_triggered_by(data):
    print("✅ Action déclenchée :", data)
    # 🔄 On retransmet sur le serveur local
    await sio_local.emit("action_triggered_by", data)




# Logs de connexion sur le serveur local ---
@sio_local.event
async def connect(sid, environ):
    print(f"🟢 Client local connecté : {sid}")

@sio_local.event
async def disconnect(sid):
    print(f"🔴 Client local déconnecté : {sid}")

async def index(request):
    return web.Response(text="✅ Serveur local Socket.IO en marche")
app.router.add_get("/", index)


# --- Boucle principale ---
async def main():
    # 🔌 Lancer la connexion au serveur Node.js
    asyncio.create_task(sio_remote.connect(REMOTE_SERVER_URL, transports=["websocket"]))

    # 🌍 Lancer le serveur Socket.IO local
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 5000)
    print("🚀 Serveur local WebSocket sur http://localhost:5000")
    await site.start()

    await sio_remote.wait()  # reste connecté et écoute en permanence

if __name__ == "__main__":
    asyncio.run(main())
