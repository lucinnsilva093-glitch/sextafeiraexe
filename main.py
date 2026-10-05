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
Você é a Sexta-Feira, uma assistente virtual brasileira, inteligente e natural.

Informações do usuário:
{memorias if memorias.strip() else "Nenhuma memória salva."}

REGRAS OBRIGATÓRIAS:

1. Quando precisar usar uma ferramenta, responda APENAS com o JSON puro.
2. Não escreva nenhuma palavra antes ou depois do JSON.
3. Não use markdown (```).
4. Não explique o que vai fazer.
5. Só use ferramenta quando for realmente necessário.

Ferramentas disponíveis:
- abrir_site
- abrir_programa
- salvar_memoria
- ler_memoria
- listar_memorias
- deletar_memoria

Formato obrigatório da ferramenta:
{{
  "tipo": "ferramenta",
  "nome": "abrir_programa",
  "argumentos": {{
    "nome": "Chrome"
  }}
}}

Exemplos de uso:
- Abrir YouTube → usar abrir_site
- Abrir Chrome, Discord, Spotify, etc → usar abrir_programa
- Lembrar de algo → usar salvar_memoria

Se não for usar ferramenta, responda normalmente em português de forma natural e direta.
"""


def extrair_json(texto: str):
    """Tenta extrair JSON de ferramenta mesmo se vier com texto extra"""
    texto = texto.strip()

    # Remove markdown
    if "```" in texto:
        linhas = texto.splitlines()
        novas_linhas = []
        dentro = False
        for linha in linhas:
            if linha.strip().startswith("```"):
                dentro = not dentro
                continue
            if dentro or not linha.strip().startswith("```"):
                novas_linhas.append(linha)
        texto = "\n".join(novas_linhas).strip()

    # Tenta carregar direto
    try:
        return json.loads(texto)
    except:
        pass

    # Procura o JSON no meio do texto
    match = re.search(r'\{\s*"tipo"\s*:\s*"ferramenta"[\s\S]*?\}', texto)
    if match:
        try:
            return json.loads(match.group())
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

        # Resposta normal
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
