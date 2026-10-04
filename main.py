from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
import os
import json
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
Você é Sexta-Feira, uma assistente virtual inteligente, educada, objetiva e útil.
Seu objetivo é ajudar o usuário da melhor forma possível.

Informações conhecidas sobre o usuário:
{memorias}

Você possui acesso às seguintes ferramentas:
- abrir_site
- abrir_programa
- salvar_memoria
- ler_memoria
- listar_memorias
- deletar_memoria

Quando precisar usar uma ferramenta, responda SOMENTE com JSON válido neste formato:
{{
  "tipo": "ferramenta",
  "nome": "nome_da_ferramenta",
  "argumentos": {{
    "chave": "valor"
  }}
}}

Regras:
- Se for usar ferramenta: responda APENAS o JSON, sem nenhum texto extra.
- Se não precisar de ferramenta: responda normalmente em português.
- Nunca invente resultados de ferramentas.
"""


@app.get("/")
def home():
    return {"status": "Sexta-Feira API online"}


@app.post("/conversar")
def conversar(req: MensagemRequest):
    try:
        historico = carregar_historico(req.session_id, limite=15)
        
        messages = [{"role": "system", "content": montar_system_prompt(req.session_id)}]
        messages.extend(historico)
        messages.append({"role": "user", "content": req.mensagem})

        resposta = client.chat.completions.create(
            model=MODELO,
            messages=messages,
            max_tokens=500
        )

        conteudo = resposta.choices[0].message.content or ""

        # Tenta detectar se é ferramenta
        try:
            dados = json.loads(conteudo.strip())
            if isinstance(dados, dict) and dados.get("tipo") == "ferramenta":
                return {
                    "tipo": "ferramenta",
                    "nome": dados.get("nome"),
                    "argumentos": dados.get("argumentos", {}),
                    "raw": conteudo
                }
        except:
            pass

        # Resposta normal
        agora = datetime.now().isoformat()
        salvar_mensagem(req.session_id, "user", req.mensagem, agora)
        salvar_mensagem(req.session_id, "assistant", conteudo, agora)

        return {
            "tipo": "texto",
            "resposta": conteudo
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool-result")
def tool_result(req: ToolResultRequest):
    """Depois que o PC executa a ferramenta, manda o resultado de volta"""
    try:
        historico = carregar_historico(req.session_id, limite=15)

        messages = [{"role": "system", "content": montar_system_prompt(req.session_id)}]
        messages.extend(historico)
        messages.append({"role": "user", "content": req.original_message})
        messages.append({
            "role": "user",
            "content": f"A ferramenta retornou:\n{req.tool_result}\n\nResponda naturalmente ao usuário com base nesse resultado."
        })

        resposta = client.chat.completions.create(
            model=MODELO,
            messages=messages,
            max_tokens=500
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
        raise HTTPException(status_code=500, detail=str(e))
