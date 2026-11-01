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

@app.get("/cerebro", response_class=HTMLResponse)
def cerebro():
    return """
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Sexta-Feira — Cérebro</title>
  <style>
    body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#0a0e17;color:#d0e8ff}
    header{padding:20px 24px;border-bottom:1px solid #123040;display:flex;justify-content:space-between;align-items:center}
    h1{margin:0;color:#00d4ff;font-size:22px}
    .status{color:#00ff9f}
    main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}
    section{background:#0d1520;border:1px solid #1f3a4d;border-radius:12px;padding:14px;min-height:50vh}
    h2{margin:0 0 12px;color:#00d4ff;font-size:16px}
    pre{white-space:pre-wrap;word-break:break-word;background:#0a0e17;padding:12px;border-radius:8px;max-height:65vh;overflow:auto}
    button{background:#003344;color:#fff;border:1px solid #00d4ff;border-radius:8px;padding:8px 12px;cursor:pointer}
    @media(max-width:800px){main{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <header>
    <h1>Sexta-Feira · Cérebro</h1>
    <div>
      <span class="status" id="status">● ...</span>
      <button onclick="carregar()">Atualizar</button>
    </div>
  </header>
  <main>
    <section>
      <h2>Memórias</h2>
      <pre id="memorias">carregando...</pre>
    </section>
    <section>
      <h2>Histórico</h2>
      <pre id="historico">carregando...</pre>
    </section>
  </main>
  <script>
    async function carregar(){
      try{
        const s = await fetch('/');
        document.getElementById('status').textContent = s.ok ? '● Online' : '● Offline';
        const m = await fetch('/api/memorias?session_id=local').then(r=>r.json());
        const h = await fetch('/api/historico?session_id=local&limite=40').then(r=>r.json());
        document.getElementById('memorias').textContent = JSON.stringify(m.fatos || m, null, 2);
        document.getElementById('historico').textContent = JSON.stringify(h.historico || h, null, 2);
      }catch(e){
        document.getElementById('status').textContent = '● Erro';
        document.getElementById('memorias').textContent = String(e);
      }
    }
    carregar();
    setInterval(carregar, 15000);
  </script>
</body>
</html>
"""

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
