import os

from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi import FastAPI, Request
from linebot import LineBotApi
from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextSendMessage

from src.rag import retrieval_augmented_generation

load_dotenv(verbose=True)

app = FastAPI()
line_bot_api = LineBotApi(os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ.get("LINE_CHANNEL_SECRET"))


@app.get("/")
def root():
    return {"message": "Welcome to LINE Bot"}


@app.post("/callback")
async def callback(
    request: Request,
) -> str:
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid Signature Error.")
    return "OK"


@handler.add(MessageEvent)
def handle_message(event: MessageEvent) -> None:
    """
    LINE Messaging APIのハンドラより呼び出される処理です。
    受け取ったメッセージに従い返信メッセージを返却します。

    Parameters
    ----------
    event : MessageEvent
        送信されたメッセージの情報です。
    """
    message = event.message.text
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=retrieval_augmented_generation.main("src/data", message)["result"]
        ),
    )
