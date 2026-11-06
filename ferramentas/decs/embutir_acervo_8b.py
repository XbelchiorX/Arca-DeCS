# -*- coding: utf-8 -*-
"""Vetores do Qwen3-Embedding-8B (texto INTEIRO: titulo + palavras-chave + resumo) para o acervo todo, para o
modelo de galho (pedido do responsavel, 2026-10-17). So calcula e guarda no cache (CuradoriaARCA/cache/
emb_itens_qwen3-8b.pt, salvo a cada 4.000; se parar, continua de onde parou). Nada e gravado no ARCA.
Uso: python ferramentas/decs/embutir_acervo_8b.py
"""
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
os.environ["DECS_MODELO_EMBEDDING"] = "qwen3-8b"
from arca import decs_acervo, decs_embeddings, decs_indexador as IX  # noqa: E402


def main():
    t0 = time.time()
    acervo = decs_acervo.carregar_acervo()
    hs = sorted(acervo)
    textos = [decs_acervo.texto_do_item(acervo[h], decs_embeddings.MAX_RESUMO) for h in hs]
    print("%d itens; cache: %s" % (len(hs), IX.ARQ_EMB_ITENS), flush=True)
    decs_embeddings.embutir_com_cache(hs, textos, IX.ARQ_EMB_ITENS)
    decs_embeddings.liberar()
    print("FIM em %.1f h" % ((time.time() - t0) / 3600), flush=True)


if __name__ == "__main__":
    main()
