from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
import os
import json
import re
from database import (
    salvar_mensagem,
    carregar_historico,
    salvar_fato,
    buscar_fatos,
    deletar_fato
)

app = FastAPI(title="Sexta-Feira API")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODELO = "deepseek/deepseek-chat-v3-0324"


class MensagemRequest(BaseModel):
    mensagem: str
    session_id: str = "local"


class ToolResultRequest(BaseModel):
    session_id: str = "local"
    tool_name: str
    tool_result: str
    original_message: str


def montar_system_prompt(session_id: str) -> str:
    fatos = buscar_fatos(session_id)
    memorias = ""
    for chave, valor in fatos.items():
        memorias += f"{chave}: {valor}\n"

    return f"""
Você é a Sexta-Feira, uma assistente virtual brasileira.

Informações do usuário:
{memorias if memorias.strip() else "Nenhuma memória salva."}

REGRAS IMPORTANTES:

REGRAS OBRIGATÓRIAS (não desobedeça):

1. Quando for usar ferramenta, você DEVE responder SOMENTE com o JSON.
2. É PROIBIDO escrever qualquer texto antes ou depois do JSON.
3. É PROIBIDO usar markdown (```).
4. É PROIBIDO explicar, dar passos ou fazer perguntas quando for usar ferramenta.
5. Se precisar usar ferramenta, a resposta deve ser APENAS o JSON puro.

Exemplo CORRETO:
{
  "tipo": "ferramenta",
  "nome": "espelhar_tela",
  "argumentos": {
    "nome": "a03core"
  }
}

Exemplo ERRADO:
Claro! Vou espelhar a tela.
{
  "tipo": "ferramenta",
  ...
}

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
- salvar_memoria

Formato obrigatório da ferramenta:
{{
  "tipo": "ferramenta",
  "nome": "nome_da_ferramenta",
  "argumentos": {{
    "chave": "valor"
  }}
}}

Exemplos de uso:

- "Abre o Spotify" → abrir_programa
- "Abre o YouTube" → abrir_site
- "Tira um print" → tirar_print
- "Fecha todos os apps" → fechar_todos_apps
- "Aumenta o volume" → controlar_volume
- "Trava o PC" → controlar_pc
- "Abre a pasta de downloads" → abrir_pasta
- "Conecta no A03 Core" → conectar_dispositivo
- "Mostra a tela do celular" → espelhar_tela
- "Espelha o A03" → espelhar_tela
- "Quais dispositivos estão conectados?" → listar_dispositivos
"""


def extrair_json(texto: str):
    import re
    import json

    if not texto:
        return None

    texto = texto.strip()

    # 1. Tenta encontrar qualquer bloco ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', texto)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # 2. Tenta encontrar JSON puro com "tipo": "ferramenta"
    match = re.search(r'\{\s*"tipo"\s*:\s*"ferramenta"[\s\S]*?\}', texto)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # 3. Última tentativa: carregar o texto inteiro
    try:
        return json.loads(texto)
    except:
        pass

    return None


@app.get("/")
def home():
    return {"status": "Sexta-Feira API online"}


@app.post("/conversar")
def conversar(req: MensagemRequest):
    try:
        historico = carregar_historico(req.session_id, limite=12)

        messages = [{"role": "system", "content": montar_system_prompt(req.session_id)}]
        messages.extend(historico)
        messages.append({"role": "user", "content": req.mensagem})

        resposta = client.chat.completions.create(
            model=MODELO,
            messages=messages,
            max_tokens=500,
            temperature=0.3
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
        historico = carregar_historico(req.session_id, limite=12)

        messages = [{"role": "system", "content": montar_system_prompt(req.session_id)}]
        messages.extend(historico)
        messages.append({"role": "user", "content": req.original_message})
        messages.append({
            "role": "user",
            "content": f"A ferramenta retornou o seguinte resultado:\n{req.tool_result}\n\nResponda naturalmente ao usuário com base nesse resultado. Não mencione JSON nem ferramentas."
        })

        resposta = client.chat.completions.create(
            model=MODELO,
            messages=messages,
            max_tokens=400,
            temperature=0.4
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
