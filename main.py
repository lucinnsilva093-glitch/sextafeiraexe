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
MODELO_VISAO = "openai/gpt-4o-mini"  # modelo com visão


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

Formato obrigatório da ferramenta:
{{
  "tipo": "ferramenta",
  "nome": "analisar_tela",
  "argumentos": {{}}
}}

Use a ferramenta analisar_tela quando o usuário pedir para:
- analisar a tela
- dar opinião sobre o que está fazendo
- ajudar com código/erro na tela
- dar dica de estudo ou jogo
"""


def extrair_json(texto: str):
    if not texto:
        return None

    texto = texto.strip()

    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', texto)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    match = re.search(r'\{\s*"tipo"\s*:\s*"ferramenta"[\s\S]*?\}', texto)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

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


@app.post("/analisar-tela")
def analisar_tela(req: AnalisarTelaRequest):
    try:
        prompt_visao = """
Você é a Sexta-Feira, uma assistente prestativa.

Analise a imagem da tela do usuário e responda em português, de forma natural e direta.

Foque em detectar dificuldades em:
- Programação / erros de código
- Estudo / exercícios
- Jogos

Se perceber dificuldade, dê uma opinião ou dica útil.
Se estiver tudo normal, apenas comente o que está vendo de forma amigável.
Não seja longa demais. Máximo 3 a 4 frases.
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
