#!/usr/bin/env python3
"""Resumen diario de noticias (tecnología, IA, ciberseguridad) por Telegram,
con selección, resumen e impacto generados por Gemini.

Variables de entorno:
  TELEGRAM_TOKEN    token del bot (@BotFather)
  TELEGRAM_CHAT_ID  tu chat id
  GEMINI_API_KEY    clave gratuita de https://aistudio.google.com/apikey
  GEMINI_MODEL      (opcional) modelo a usar, por defecto gemini-3.6-flash

Dependencias: pip install feedparser requests
"""
import html
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests

HORAS_VENTANA = 30          # solo noticias de las últimas N horas
CANDIDATAS_POR_SECCION = 15  # las que se envían a Gemini para que elija
NOTICIAS_FINALES = 10        # las que llegan a Telegram
MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

FEEDS = {
    "Tecnología": [
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://techcrunch.com/feed/",
    ],
    "IA": [
        "https://www.technologyreview.com/feed/",
        "https://venturebeat.com/category/ai/feed/",
        "https://huggingface.co/blog/feed.xml",
    ],
    "Ciberseguridad": [
        "https://feeds.feedburner.com/TheHackersNews",
        "https://www.bleepingcomputer.com/feed/",
        "https://krebsonsecurity.com/feed/",
        "https://www.darkreading.com/rss.xml",
    ],
}
ICONOS = {"Tecnología": "💻", "IA": "🤖", "Ciberseguridad": "🔐"}

PALABRAS_CLAVE = {
    "zero-day": 3, "ransomware": 3, "vulnerabilidad": 3, "vulnerability": 3,
    "breach": 3, "brecha": 3, "cve-": 2, "exploit": 2, "actively exploited": 3,
    "openai": 2, "anthropic": 2, "claude": 2, "gemini": 2, "gpt": 2,
    "llm": 2, "nvidia": 1, "regulation": 2, "regulación": 2, "ai act": 2,
    "launch": 1, "lanza": 1, "acquisition": 1, "adquiere": 1,
}


# ---------- Recogida de noticias ----------

def limpiar(texto: str, max_len: int = 300) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto or "")
    texto = html.unescape(re.sub(r"\s+", " ", texto)).strip()
    return texto if len(texto) <= max_len else texto[: max_len - 1].rsplit(" ", 1)[0] + "…"


def fecha_entrada(e):
    t = e.get("published_parsed") or e.get("updated_parsed")
    return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc) if t else None


def prepuntuar(titulo, resumen, horas):
    texto = f"{titulo} {resumen}".lower()
    score = sum(p for k, p in PALABRAS_CLAVE.items() if k in texto)
    return score + max(0, (HORAS_VENTANA - horas) / HORAS_VENTANA)


def recoger_candidatas():
    limite = datetime.now(timezone.utc) - timedelta(hours=HORAS_VENTANA)
    candidatas, vistos = [], set()
    for seccion, urls in FEEDS.items():
        items = []
        for url in urls:
            try:
                feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            except Exception as exc:
                print(f"[WARN] {url}: {exc}")
                continue
            fuente = feed.feed.get("title", url)
            for e in feed.entries:
                f, link = fecha_entrada(e), e.get("link")
                if not f or f < limite or not link or link in vistos:
                    continue
                vistos.add(link)
                titulo = limpiar(e.get("title", ""), 160)
                resumen = limpiar(e.get("summary", ""))
                horas = (datetime.now(timezone.utc) - f).total_seconds() / 3600
                items.append({
                    "categoria": seccion, "fuente": fuente, "titulo": titulo,
                    "extracto": resumen, "link": link,
                    "score": prepuntuar(titulo, resumen, horas),
                })
        items.sort(key=lambda x: x["score"], reverse=True)
        candidatas.extend(items[:CANDIDATAS_POR_SECCION])
    for i, c in enumerate(candidatas):
        c["id"] = i
    return candidatas


# ---------- Gemini ----------

PROMPT = """Eres un analista de tecnología, IA y ciberseguridad. Te paso una lista de noticias candidatas de las últimas 24 horas (JSON).

Tareas:
1. Elige las {n} más importantes e influyentes para un comercial del sector IT que quiere identificar noticias que puedan ayudarle a posicionar servicios de advisory tecnológico estratégico (IA, ciberseguridad, estrategia IT, vendors, etc.). Si varias hablan de lo mismo, quédate con una sola.
2. Ordénalas de MAYOR a MENOR importancia.
3. Para cada una escribe, en español: un título breve, un resumen de unas 100 palabras y el impacto (1-2 frases: a quién afecta y por qué importa).

No inventes datos que no aparezcan en el título o extracto. Responde SOLO con un array JSON con objetos de esta forma:
{{"id": <id de la noticia>, "titulo": "...", "resumen": "...", "impacto": "..."}}

Noticias:
{noticias}
"""


def llamar_gemini(candidatas):
    clave = os.environ.get("GEMINI_API_KEY")
    if not clave:
        raise RuntimeError("Falta GEMINI_API_KEY (secret no configurado o workflow sin la variable)")
    entrada = [
        {k: c[k] for k in ("id", "categoria", "fuente", "titulo", "extracto")}
        for c in candidatas
    ]
    cuerpo = {
        "contents": [{"parts": [{"text": PROMPT.format(
            n=NOTICIAS_FINALES, noticias=json.dumps(entrada, ensure_ascii=False))}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO}:generateContent"
    for intento in range(4):
        r = requests.post(url, json=cuerpo, headers={"x-goog-api-key": clave}, timeout=120)
        if r.status_code in (429, 500, 503):
            time.sleep(10 * (intento + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"Gemini HTTP {r.status_code} (modelo {MODELO}): {r.text[:300]}")
        texto = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto.strip())
        return json.loads(texto)
    raise RuntimeError(f"Gemini sin respuesta tras reintentos (HTTP {r.status_code}): {r.text[:300]}")


# ---------- Telegram ----------

def enviar(token, chat_id, texto):
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": texto, "parse_mode": "HTML",
              "disable_web_page_preview": True},
        timeout=30,
    )
    r.raise_for_status()


def bloque(pos, noticia, base):
    esc = html.escape
    icono = ICONOS.get(base["categoria"], "📰")
    return (
        f"<b>{pos}. {esc(noticia['titulo'])}</b>\n"
        f"{icono} {esc(base['categoria'])} · <i>{esc(base['fuente'])}</i>\n"
        f"{esc(noticia['resumen'])}\n"
        f"💥 <b>Impacto:</b> {esc(noticia['impacto'])}\n"
        f"🔗 <a href=\"{esc(base['link'])}\">Leer noticia</a>"
    )


def bloque_sin_ia(pos, base):
    esc = html.escape
    return (
        f"<b>{pos}. {esc(base['titulo'])}</b>\n"
        f"{ICONOS.get(base['categoria'], '📰')} {esc(base['categoria'])} · "
        f"<i>{esc(base['fuente'])}</i>\n"
        f"🔗 <a href=\"{esc(base['link'])}\">Leer noticia</a>"
    )


def enviar_en_tandas(token, chat_id, cabecera, bloques):
    """Agrupa bloques en mensajes que no superen el límite de 4096 caracteres."""
    actual = cabecera
    for b in bloques:
        if len(actual) + len(b) + 2 > 3900:
            enviar(token, chat_id, actual)
            actual = ""
        actual += ("\n\n" if actual else "") + b
    if actual.strip():
        enviar(token, chat_id, actual)


def main():
    token, chat_id = os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    candidatas = recoger_candidatas()
    if not candidatas:
        enviar(token, chat_id, "Hoy no he encontrado noticias nuevas en los feeds.")
        return

    por_id = {c["id"]: c for c in candidatas}
    cabecera = f"☀️ <b>Resumen del {datetime.now():%d/%m/%Y}</b> (por importancia)"

    try:
        seleccion = llamar_gemini(candidatas)
        bloques = []
        for n in seleccion[:NOTICIAS_FINALES]:
            base = por_id.get(n.get("id"))
            if base:  # el enlace sale siempre del feed, nunca del modelo
                bloques.append(bloque(len(bloques) + 1, n, base))
        if not bloques:
            raise ValueError("Gemini no devolvió noticias válidas")
    except Exception as exc:
        print(f"[WARN] Gemini falló, envío sin resumen: {exc}")
        enviar(token, chat_id,
               "⚠️ Gemini ha fallado, envío solo títulos.\n"
               f"<code>{html.escape(str(exc)[:500])}</code>")
        top = sorted(candidatas, key=lambda c: c["score"], reverse=True)[:NOTICIAS_FINALES]
        bloques = [bloque_sin_ia(i + 1, c) for i, c in enumerate(top)]

    enviar_en_tandas(token, chat_id, cabecera, bloques)


if __name__ == "__main__":
    main()
