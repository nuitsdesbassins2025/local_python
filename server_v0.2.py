import socketio # type: ignore
import asyncio
from aiohttp import web

REMOTE_SERVER_URL = "https://nuit-des-bassins-client-9b7778c21473.herokuapp.com/"

REMOTE_SERVER_URL = "http://localhost:3001/"

sio_remote = socketio.AsyncClient()
sio_local = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
app = web.Application()
sio_local.attach(app)

# Utiliser un Event pour contrôler l'arrêt du script
shutdown_event = asyncio.Event()

@sio_remote.event
async def connect():
    print("✅ Connecté au serveur Node.js")
    await sio_remote.emit("client_request_datas", {"client_id": "id-admin1234"})
    await sio_remote.emit("send_message", {"target": "all", "message": "admin connecté !", "notification": False})

    # await sio_remote.emit("client_update_datas", {"client_id": "id-admin1234", "pseudo": "leo_admin"})

@sio_remote.event
async def disconnect():
    print("❌ Déconnecté du serveur Node.js")

@sio_remote.on("user_data")
async def on_user_data(data):
    print("📦 Données utilisateur reçues :", data)

@sio_remote.on("emit_message")
async def on_emit_message(data):
    print("📢 Nouveau message reçu :", data)

@sio_remote.on("pseudo_updated")
async def on_pseudo_updated(data):
    print("📝 Pseudo mis à jour :", data)

@sio_remote.on("continuous_data")
async def on_continuous_data(data):
    print("📝 Data mises à jour :", data)

@sio_remote.on("action_triggered_by")
async def on_action_triggered_by(data):
    print("✅ Action déclenchée :", data)
    await sio_local.emit("action_triggered_by", data)

@sio_remote.on("admin_game_setting")
async def on_action_triggered_by(data):
    print(data)
    print("✅ changement de scene demandé :", data)
    await sio_local.emit("admin_game_setting", data)
    #  await sio_local.emit(  "admin_game_setting", {client_id, action, value })


# [{ detection_id , player_id , posX , posY}]

# client_in_aperage <= emit depuis le serveur web  : si joueur présent dans la zone d'apérage : associer client_id et detection_id et player_id


@sio_local.on("gd_ball_bounce")
async def ball_bounce(id, data):
    print("✅ rebond balle :", data)
    await sio_remote.emit("ball_bounce", data)




@sio_local.event
async def connect(sid, environ):
    print(f"🟢 Client local connecté : {sid}")

@sio_local.event
async def disconnect(sid):
    print(f"🔴 Client local déconnecté : {sid}")

async def index(request):
    return web.Response(text="✅ Serveur local Socket.IO en marche")

app.router.add_get("/", index)

async def main():
    runner = None  # Déclarer runner en dehors du bloc try

    try:
        print(f"🔗 Connexion au serveur distant.: {REMOTE_SERVER_URL}")
        await sio_remote.connect(REMOTE_SERVER_URL, transports=["websocket"])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", 5000)
        print("🚀 Serveur local WebSocket sur http://localhost:5000")
        await site.start()

        # Attendre indéfiniment jusqu'à ce que l'événement d'arrêt soit déclenché
        await shutdown_event.wait()
    except KeyboardInterrupt:
        print("Arrêt du serveur demandé par l'utilisateur")
    finally:
        await sio_remote.disconnect()
        if runner is not None:  # Vérifier que runner existe avant de le nettoyer
            await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
