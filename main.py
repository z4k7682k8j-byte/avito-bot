import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, Request, BackgroundTasks

load_dotenv()

AVITO_CLIENT_ID = os.getenv("AVITO_CLIENT_ID")
AVITO_CLIENT_SECRET = os.getenv("AVITO_CLIENT_SECRET")
AVITO_USER_ID = os.getenv("AVITO_USER_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Avito AI Autoresponder")

SYSTEM_PROMPT = """
Ты — вежливый помощник продавца на Авито.
Отвечай кратко (1-3 предложения), доброжелательно и по делу на русском языке.
Если клиент спрашивает о наличии/актуальности: подтверждай, что объявление актуально.
Если клиент спрашивает о деталях, которых ты не знаешь (точный адрес, время встречи, торг): скажи, что продавец скоро подключится и ответит подробнее.
Не придумывай вымышленные адреса и номера телефонов.
"""

def get_avito_token() -> str:
    url = "https://api.avito.ru/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AVITO_CLIENT_ID,
        "client_secret": AVITO_CLIENT_SECRET,
    }
    response = requests.post(url, data=payload, timeout=10)
    response.raise_for_status()
    return response.json()["access_token"]

def ask_groq_ai(user_text: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.3,
        "max_tokens": 150,
    }
    response = requests.post(url, json=body, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def send_reply_to_avito(chat_id: str, message_text: str):
    token = get_avito_token()
    url = f"https://api.avito.ru/messenger/v2/accounts/{AVITO_USER_ID}/chats/{chat_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "message": {"text": message_text},
        "type": "text"
    }
    response = requests.post(url, json=body, headers=headers, timeout=10)
    if response.status_code in [200, 201]:
        logger.info(f"Сообщение успешно отправлено в чат {chat_id}")
    else:
        logger.error(f"Ошибка отправки в Авито ({response.status_code}): {response.text}")

def handle_message_background(event: Dict[str, Any]):
    try:
        payload = event.get("payload", {}).get("value", {})
        author_id = payload.get("author_id")
        if str(author_id) == str(AVITO_USER_ID):
            return

        chat_id = payload.get("chat_id")
        content = payload.get("content", {})
        incoming_text = content.get("text")

        if not incoming_text or not chat_id:
            return

        logger.info(f"Получено сообщение: '{incoming_text}'")
        reply_text = ask_groq_ai(incoming_text)
        logger.info(f"Ответ ИИ: '{reply_text}'")
        send_reply_to_avito(chat_id, reply_text)
    except Exception as e:
        logger.exception(f"Ошибка при обработке: {e}")

@app.post("/webhook")
async def avito_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    background_tasks.add_task(handle_message_background, data)
    return {"status": "accepted"}
