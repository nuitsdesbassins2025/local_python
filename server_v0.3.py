import socketio  # type: ignore
import asyncio
from aiohttp import web
from fastapi import FastAPI, Request
import uvicorn

REMOTE_SERVER_URL = "https://nuit-des-bassins-client-9b7778c21473.herokuapp.com/"

# REMOTE_SERVER_URL = "http://localhost:3001/"

sio_remote = socketio.AsyncClient()
sio_local = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
app = web.Application()
sio_local.attach(app)

# FastAPI
app_fastapi = FastAPI()

# Utiliser un Event pour contrôler l'arrêt du script
shutdown_event = asyncio.Event()


@sio_remote.event
async def connect():
    print("✅ Connecté au serveur Node.js")
    await sio_remote.emit("client_request_datas", {"client_id": "id-admin1234"})

    await sio_remote.emit("send_message", {"target": "all", "message": "admin connecté !", "notification": False})


@sio_remote.event
async def disconnect():
    print("❌ Déconnecté du serveur Node.js")


# @sio_remote.on("user_data")
# async def on_user_data(data):
#    print("📦 Données utilisateur reçues :", data)

@sio_remote.on("emit_message")
async def on_emit_message(data):
    print("📢 Nouveau message reçu :", data)


# @sio_remote.on("pseudo_updated")
# async def on_pseudo_updated(data):
#    print("📝 Pseudo mis à jour :", data)

# @sio_remote.on("continuous_data")
# async def on_continuous_data(data):
#    print("📝 Data mises à jour :", data)


# Va etre remplace par cliet_action_trigger
@sio_remote.on("action_triggered_by")
async def on_action_triggered_by(data):
    print("✅ Action déclenchée :", data)
    await sio_local.emit("action_triggered_by", data)


# A developper / renomer
@sio_remote.on("admin_game_setting")
async def on_action_triggered_by(data):
    print(data)
    print("✅ changement de scene demandé :", data)
    await sio_local.emit("admin_game_setting", data)
    #  await sio_local.emit(  "admin_game_setting", {client_id, action, value })


@sio_remote.on("client_action_trigger")
async def on_action_triggered(datas):
    client_id = datas.get("client_id", None)
    action = datas.get("action", None)
    action_datas = datas.get("datas", None)
    print("✅ action reçue du distant :", client_id, action, datas)

    emit_data = {
        "client_id": client_id,
        "action": action,
        "datas": action_datas
    }

    await sio_local.emit("client_action_trigger", emit_data)
    print("✅ action reçue du local :", client_id, action, action_datas)




async def on_tracking_lost(tracking_id, client_id):
    await sio_local.emit("tracking_lost", tracking_id)


async def on_tracking_recover(tracking_id, client_id):
    await sio_local.emit("tracking_recovered", tracking_id, client_id)


@sio_local.event
async def connect(sid, environ):
    print(f"🟢 Client local connecté : {sid}")


@sio_local.event
async def disconnect(sid):
    print(f"🔴 Client local déconnecté : {sid}")


async def index(request):
    return web.Response(text="✅ Serveur local Socket.IO en marche")



# ----------- FASTAPI REST API ----------------
@app_fastapi.post("/camera/detection")
async def camera_detection(request: Request):

    camera_data = await request.json()
    #print("📡 REST API: données reçues sur /camera/detection", data)

    emit_data = {
        "tracking_fps": camera_data.get('tracking_fps', 0.0),
        "tracking_datas": camera_data.get('tracking_datas', []),
    }

    await sio_local.emit("tracking_datas", emit_data)
    return {"status": "ok"}

async def start_fastapi():
    config = uvicorn.Config(app_fastapi, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    await server.serve()

# ----------- FASTAPI REST API ----------------

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
        print("🚀 Serveur local FastAPI sur http://localhost:8000, /camera/detection")
        await start_fastapi()

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
