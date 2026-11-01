from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
import os
import json
import re
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from database import (
    salvar_mensagem,
    carregar_historico,
    buscar_fatos,
)

app = FastAPI(title="Sexta-Feira API")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODELO = "openrouter/free"
MODELO_VISAO = "openai/gpt-4o-mini"

@app.get("/api/memorias")
def api_memorias(session_id: str = "local"):
    try:
        return {"session_id": session_id, "fatos": buscar_fatos(session_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/historico")
def api_historico(session_id: str = "local", limite: int = 30):
    try:
        return {
            "session_id": session_id,
            "historico": carregar_historico(session_id, limite=limite)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MensagemRequest(BaseModel):
    mensagem: str
    session_id: str = "local"


class ToolResultRequest(BaseModel):
    session_id: str = "local"
    tool_name: str
    tool_result: str
    original_message: str


class AnalisarTelaRequest(BaseModel):
    imagem_base64: str
    session_id: str = "local"


def montar_system_prompt(session_id: str) -> str:
    fatos = buscar_fatos(session_id)
    memorias = ""
    for chave, valor in fatos.items():
        memorias += f"{chave}: {valor}\n"

    return f"""
Você é a Sexta-Feira, uma assistente virtual brasileira.

Informações do usuário:
{memorias if memorias.strip() else "Nenhuma memória salva."}

REGRAS OBRIGATÓRIAS:
1. Quando for usar ferramenta, responda SOMENTE com o JSON puro.
2. É PROIBIDO escrever qualquer texto antes ou depois do JSON.
3. É PROIBIDO usar markdown.
4. Se não for usar ferramenta, responda normalmente em português.
5. Use pesquisar_web quando o usuário pedir informação atual:
notícias, preços, clima, resultados de jogos, fatos recentes, etc.

Ferramentas disponíveis:
- abrir_site
- abrir_programa
- listar_programas_abertos
- tirar_print
- controlar_volume
- controlar_pc
- abrir_pasta
- ler_area_transferencia
- fechar_todos_apps
- conectar_dispositivo
- espelhar_tela
- listar_dispositivos
- analisar_tela
- salvar_memoria
- spotify_tocando_agora
- spotify_play_pause
- spotify_proxima
- spotify_anterior
- spotify_pesquisar_tocar
- spotify_tocar_playlist
- pesquisar_web


Se pesquisar_web retornar links de clima/notícias, isso conta como sucesso.
Nunca diga que não conseguiu se houver resultados.

Formato obrigatório:
{{
  "tipo": "ferramenta",
  "nome": "nome_da_ferramenta",
  "argumentos": {{
    "chave": "valor"
  }}
}}

Exemplos:

Pesquisar na internet:
{{
  "tipo": "ferramenta",
  "nome": "pesquisar_web",
  "argumentos": {{
    "query": "previsão do tempo São Paulo hoje"
  }}
}}

Abrir programa:
{{
  "tipo": "ferramenta",
  "nome": "abrir_programa",
  "argumentos": {{
    "nome": "spotify"
  }}
}}

Aumentar volume:
{{
  "tipo": "ferramenta",
  "nome": "controlar_volume",
  "argumentos": {{
    "acao": "aumentar"
  }}
}}

Diminuir volume:
{{
  "tipo": "ferramenta",
  "nome": "controlar_volume",
  "argumentos": {{
    "acao": "diminuir"
  }}
}}

Travar PC:
{{
  "tipo": "ferramenta",
  "nome": "controlar_pc",
  "argumentos": {{
    "acao": "travar"
  }}
}}

Abrir site:
{{
  "tipo": "ferramenta",
  "nome": "abrir_site",
  "argumentos": {{
    "url": "youtube.com"
  }}
}}

Spotify - tocar música:
{{
  "tipo": "ferramenta",
  "nome": "spotify_pesquisar_tocar",
  "argumentos": {{
    "query": "Imagine Dragons"
  }}
}}

Spotify - pausar:
{{
  "tipo": "ferramenta",
  "nome": "spotify_play_pause",
  "argumentos": {{
    "acao": "pause"
  }}
}}

Spotify - próxima:
{{
  "tipo": "ferramenta",
  "nome": "spotify_proxima",
  "argumentos": {{}}
}}
"""


def extrair_json(texto: str):
    if not texto:
        return None

    texto = texto.strip()

    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", texto)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    match = re.search(r'\{\s*"tipo"\s*:\s*"ferramenta"[\s\S]*?\}', texto)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    try:
        return json.loads(texto)
    except Exception:
        pass

    return None


@app.get("/")
def home():
    return {"status": "Sexta-Feira API online"}


@app.post("/conversar")
def conversar(req: MensagemRequest):
    try:
        historico = carregar_historico(req.session_id, limite=10)

        messages = [{"role": "system", "content": montar_system_prompt(req.session_id)}]
        messages.extend(historico)
        messages.append({"role": "user", "content": req.mensagem})

        resposta = client.chat.completions.create(
            model=MODELO,
            messages=messages,
            max_tokens=700,
            temperature=0.2
        )

        conteudo = resposta.choices[0].message.content or ""
        print("Resposta bruta da IA:", conteudo)

        dados = extrair_json(conteudo)

        if dados and isinstance(dados, dict) and dados.get("tipo") == "ferramenta":
            return {
                "tipo": "ferramenta",
                "nome": dados.get("nome"),
                "argumentos": dados.get("argumentos", {}),
                "raw": conteudo
            }

        agora = datetime.now().isoformat()
        salvar_mensagem(req.session_id, "user", req.mensagem, agora)
        salvar_mensagem(req.session_id, "assistant", conteudo, agora)

        return {
            "tipo": "texto",
            "resposta": conteudo
        }

    except Exception as e:
        print("Erro:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool-result")
def tool_result(req: ToolResultRequest):
    try:
        historico = carregar_historico(req.session_id, limite=10)

        messages = [{"role": "system", "content": montar_system_prompt(req.session_id)}]
        messages.extend(historico)
        messages.append({"role": "user", "content": req.original_message})
        messages.append({
            "role": "user",
            "content": (
                f"A ferramenta retornou o seguinte resultado:\n{req.tool_result}\n\n"
                "Use ESSE resultado para responder. "
                "Não diga que a busca falhou se houver links ou trechos. "
                "Resuma em português de forma natural. "
                "Se for previsão do tempo e só houver sites, cite os sites e ofereça abrir um. "
                "Não mencione JSON nem ferramentas."
            )
        })

        resposta = client.chat.completions.create(
            model=MODELO,
            messages=messages,
            max_tokens=400,
            temperature=0.3
        )

        conteudo = resposta.choices[0].message.content or ""

        agora = datetime.now().isoformat()
        salvar_mensagem(req.session_id, "user", req.original_message, agora)
        salvar_mensagem(req.session_id, "assistant", conteudo, agora)

        return {
            "tipo": "texto",
            "resposta": conteudo
        }

    except Exception as e:
        print("Erro no tool-result:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analisar-tela")
def analisar_tela(req: AnalisarTelaRequest):
    try:
        prompt_visao = """
Você é a Sexta-Feira, uma assistente prestativa.
Analise a imagem da tela do usuário e responda em português, de forma natural e direta.
Foque em dificuldades de programação, estudo ou jogos.
Se estiver tudo normal, comente de forma breve.
Máximo 3 a 4 frases.
"""

        resposta = client.chat.completions.create(
            model=MODELO_VISAO,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_visao},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{req.imagem_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.4
        )

        conteudo = resposta.choices[0].message.content or "Não consegui analisar a tela."

        agora = datetime.now().isoformat()
        salvar_mensagem(req.session_id, "user", "[pedido de análise de tela]", agora)
        salvar_mensagem(req.session_id, "assistant", conteudo, agora)

        return {
            "tipo": "texto",
            "resposta": conteudo
        }

    except Exception as e:
        print("Erro na análise de tela:", e)
        raise HTTPException(status_code=500, detail=str(e))
