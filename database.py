import os
from supabase import create_client, Client
from typing import List, Dict, Optional

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL e SUPABASE_KEY não configurados")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def salvar_mensagem(session_id: str, role: str, content: str, timestamp: str) -> None:
    supabase.table("mensagens").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": timestamp
    }).execute()


def carregar_historico(session_id: str, limite: int = 20) -> List[Dict[str, str]]:
    response = (
        supabase.table("mensagens")
        .select("role, content")
        .eq("session_id", session_id)
        .order("id", desc=True)
        .limit(limite)
        .execute()
    )
    dados = response.data or []
    dados.reverse()
    return [{"role": item["role"], "content": item["content"]} for item in dados]


def salvar_fato(session_id: str, chave: str, valor: str) -> None:
    supabase.table("memoria_importante").upsert(
        {
            "session_id": session_id,
            "chave": chave,
            "valor": valor
        },
        on_conflict="session_id,chave"
    ).execute()


def buscar_fatos(session_id: str) -> Dict[str, str]:
    response = (
        supabase.table("memoria_importante")
        .select("chave, valor")
        .eq("session_id", session_id)
        .execute()
    )
    dados = response.data or []
    return {item["chave"]: item["valor"] for item in dados}


def deletar_fato(session_id: str, chave: str) -> None:
    (
        supabase.table("memoria_importante")
        .delete()
        .eq("session_id", session_id)
        .eq("chave", chave)
        .execute()
    )
