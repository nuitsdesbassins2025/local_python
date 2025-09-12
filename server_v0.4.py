import socketio  # type: ignore
import asyncio
from aiohttp import web
from fastapi import FastAPI, Request
import uvicorn

import os
import json
import time


MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)


REMOTE_SERVER_URL = "https://nuit-des-bassins-client-9b7778c21473.herokuapp.com/"

#REMOTE_SERVER_URL = "http://localhost:3000/"

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






# A developper / renomer
@sio_remote.on("admin_game_setting")
async def on_action_triggered_by(data):
    print(data)
    print("✅ changement de scene demandé :", data)
    await sio_local.emit("admin_game_setting", data)
    #  await sio_local.emit(  "admin_game_setting", {client_id, action, value })




@sio_remote.on("client_action_trigger")
async def on_action_triggered(datas):
    # client_id = datas.get("client_id", None)
    # action = datas.get("action", None)
    # action_datas = datas.get("datas", None)
    # player_id = datas.get("player_id", None)

    # print("☎️ action reçue du distant :",  datas)

    # emit_data = {
    #     "client_id":  datas.get("client_id", None),
    #     "client_datas": datas.get("client_datas", {}),
    #     "action": datas.get("action", None),
    #     "datas": datas.get("datas", None),
    #     "player_id": datas.get("player_id", None)
    # }
    await sio_local.emit("client_action_trigger", datas)
    print("📞 'client_action_trigger' transmise au local :", datas)


@sio_remote.on("web_client_updated")
async def on_web_client_updated(updated_datas):
    print("✅ web_client_updated reçu du distant :", updated_datas)

    await sio_local.emit("web_client_updated", updated_datas)

    print("📞 'web_client_updated' transmise au local :", updated_datas)


@sio_remote.on("admin_game_settings")
async def on_admin_game_settings(settings_datas):
    print("✅ admin_game_settings reçu du distant :", settings_datas)

    await sio_local.emit("admin_game_settings", settings_datas)

    print("📞 'admin_game_settings' transmise au local :", settings_datas)


@sio_local.on("godot_event")
async def on_godot_event(signal_id, datas):
    #print("✅ evènement reçue de godot :", datas)
    event_type = datas.get("event_type", None)
    #action = datas.get("action", None)
    event_datas = datas.get("event_datas", None)

    print("✅ Godot event_type :", event_type, ", event_datas : ", event_datas)

    emit_data = {
        "event_type": event_type,
        "event_datas": event_datas
    }

    await sio_remote.emit("godot_event_transfer", emit_data)
    await sio_local.emit("godot_event_transfer", emit_data)
    #print("✅ action reçue du local :", event_type, action, event_datas)



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


def extension_from_mime(mime: str) -> str:
    """
    Retourne une extension basée sur le mimeType
    """
    if not mime:
        return ".bin"
    mime = mime.lower()
    if "webm" in mime:
        return ".webm"
    if "ogg" in mime:
        return ".ogg"
    if "wav" in mime:
        return ".wav"
    return ".bin"


@sio_remote.on("clients_media")
async def on_clients_media(data):
    """
    data = {
        "emiter": client_id,
        "media_type": "sound_graph" | "sound_track",
        "media_data": { ... }
    }
    """
    client_id = data.get("emiter", "unknown")
    media_type = data.get("media_type")
    media_data = data.get("media_data")

    # Timestamp à la seconde
    timestamp = int(time.time())

    if media_type == "sound_graph":
        filename = f"{client_id}_{timestamp}.json"
        filepath = os.path.join(MEDIA_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(media_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Graphe sonore sauvegardé : {filepath}")

    elif media_type == "sound_track":
        mime = media_data.get("mimeType", "")
        ext = extension_from_mime(mime)

        filename = f"{client_id}_{timestamp}{ext}"
        filepath = os.path.join(MEDIA_DIR, filename)

        buffer = media_data.get("buffer")
        if isinstance(buffer, list):  # Uint8Array → list[int]
            buffer = bytes(buffer)

        with open(filepath, "wb") as f:
            f.write(buffer)

        print(f"💾 Fichier audio sauvegardé : {filepath} ({mime})")

    else:
        print(f"⚠️ Type média inconnu : {media_type}")





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

