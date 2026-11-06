# -*- coding: utf-8 -*-
"""Embeddings de texto (documentos e descritores).

Um espaco vetorial so, multilingue, para tudo. O pipeline antigo usava dois
modelos (BrazEmbed para PT, e5 para EN/ES/FR), o que deixava itens de idiomas
diferentes em espacos diferentes: um artigo em ingles nunca era "vizinho" de
uma tese em portugues sobre o mesmo tema. Para o kNN isso e fatal.

O modelo e escolhido em arca/config.py (MODELO_EMBEDDING), entre os
registrados em MODELOS. Uma COMBINACAO ("e5-large+qwen3-0.6b") concatena os
vetores normalizados de cada modelo e normaliza de novo: a similaridade passa
a ser a media das similaridades de cada modelo. Trocar de modelo exige refazer
os embeddings (automatico: o cache tem o nome do modelo) e retreinar (D5).
"""
import os

import torch

from . import config

MODELOS = {
    # 2023; 19o de 93 no MTEB-BR (2026). Convencao do e5: "query: " nos dois lados.
    "e5-large": {"hf": "intfloat/multilingual-e5-large", "pooling": "media",
                 "prefixo": "query: "},
    # 2025; familia Qwen3-Embedding lidera o MTEB multilingue e e 2a no MTEB-BR
    # (o 8B). O 0.6B cabe numa GPU de 8 GB. Pooling pelo ultimo token, com
    # instrucao (o modelo foi treinado assim).
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-Embedding-0.6B", "pooling": "ultimo",
                   "prefixo": "Instruct: Identify the health and biomedical subjects of this "
                              "scientific document\nQuery: "},
    # 2025; 2o de 93 no MTEB-BR (2026), o mesmo do ODS (D17). GGUF Q8_0 no llama-server (arca/ods_metabusca):
    # le o texto INTEIRO (o servidor recusa o que passar do lote, nunca corta). Decisao do responsavel
    # (05/10): testar no DeCS; so vira padrao depois de medido em ferramentas/decs/treinar_avaliar.py.
    "qwen3-8b": {"gguf": True, "pooling": "ultimo",
                 "prefixo": "Instruct: Identify the health and biomedical subjects of this "
                            "scientific document\nQuery: "},
}
PARTES = config.MODELO_EMBEDDING.split("+")
for _p in PARTES:
    if _p not in MODELOS:
        raise ValueError("MODELO_EMBEDDING desconhecido: %s" % _p)
MODELO = config.MODELO_EMBEDDING
MAX_TOKENS = 512
# Resumo inteiro quando todos os modelos leem texto longo (regra "nada cortado"); o e5 so le 512 tokens,
# entao com ele o resumo continua limitado (o que vem primeiro e o que sobrevive, ver decs_acervo).
MAX_RESUMO = None if all(MODELOS[p].get("gguf") for p in PARTES) else 2500
LOTE = 32   # 64 estourou os 8 GB da GPU desta maquina (paginacao do Windows: travou)

_cache = {}


def _carregar(chave):
    if chave not in _cache:
        from transformers import AutoModel, AutoTokenizer
        esp = MODELOS[chave]
        tok = AutoTokenizer.from_pretrained(
            esp["hf"], padding_side="left" if esp["pooling"] == "ultimo" else "right")
        mdl = AutoModel.from_pretrained(esp["hf"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = (mdl.half() if dev == "cuda" else mdl).to(dev).eval()
        _cache[chave] = (tok, mdl, dev)
    return _cache[chave]


def liberar():
    """Libera a GPU."""
    _cache.clear()
    if any(MODELOS[p].get("gguf") for p in PARTES):
        from . import ods_metabusca as MB
        MB.liberar()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _embutir_um(chave, textos, mostrar_progresso=False):
    """Textos -> tensor float16 normalizado (n, dim) com UM modelo, na CPU.

    Ordena por tamanho antes de montar os lotes (menos padding = 3-4x mais
    rapido) e devolve na ordem original.
    """
    esp = MODELOS[chave]
    if esp.get("gguf"):
        from . import ods_metabusca as MB
        v = MB._encode([esp["prefixo"] + (t or "") for t in textos])
        return torch.from_numpy(v).half()
    tok, mdl, dev = _carregar(chave)
    ordem = sorted(range(len(textos)), key=lambda i: len(textos[i] or ""))
    saida = torch.empty((len(textos), mdl.config.hidden_size), dtype=torch.float16)
    for ini in range(0, len(ordem), LOTE):
        idx = ordem[ini:ini + LOTE]
        bloco = [esp["prefixo"] + (textos[i] or "") for i in idx]
        enc = tok(bloco, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_TOKENS).to(dev)
        h = mdl(**enc).last_hidden_state
        if esp["pooling"] == "ultimo":
            v = h[:, -1]                      # padding a esquerda: ultimo token real
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        v = torch.nn.functional.normalize(v.float(), dim=1)
        saida[idx] = v.half().cpu()
        if mostrar_progresso and (ini // LOTE) % 50 == 0:
            print("  embeddings: %d/%d" % (min(ini + LOTE, len(textos)), len(textos)), flush=True)
    return saida


def _combinar(vetores):
    v = torch.cat([x.float() for x in vetores], dim=1)
    return torch.nn.functional.normalize(v, dim=1).half()


def embutir(textos, mostrar_progresso=False):
    """Textos -> tensor float16 normalizado com o modelo (ou combinacao) configurado."""
    return _combinar([_embutir_um(c, textos, mostrar_progresso) for c in PARTES])


def embutir_com_cache(chaves, textos, caminho, mostrar_progresso=True, bloco=4000):
    """Embeddings com cache em disco, reaproveitando o que nao mudou.

    `caminho` tem o nome do modelo configurado. Numa combinacao, cada parte usa
    o seu proprio cache (o mesmo nome com a parte no lugar da combinacao) e o
    resultado e concatenado — nada e recalculado a toa.
    """
    if len(PARTES) > 1:
        partes = [_cache_de_um(p, chaves, textos, caminho.replace(MODELO, p), mostrar_progresso, bloco)
                  for p in PARTES]
        return _combinar(partes)
    return _cache_de_um(PARTES[0], chaves, textos, caminho, mostrar_progresso, bloco)


def _cache_de_um(chave, chaves, textos, caminho, mostrar_progresso, bloco):
    """O trabalho e salvo a cada `bloco` textos: se cair, continua de onde parou."""
    import time
    if os.path.exists(caminho):
        antigo = torch.load(caminho)
        mapa = {k: (t, antigo["vetores"][i]) for i, (k, t) in
                enumerate(zip(antigo["chaves"], antigo["textos"]))}
    else:
        mapa = {}
    faltam = [i for i, k in enumerate(chaves) if k not in mapa or mapa[k][0] != textos[i]]
    t0 = time.time()
    for ini in range(0, len(faltam), bloco):
        parte = faltam[ini:ini + bloco]
        vet = _embutir_um(chave, [textos[i] for i in parte])
        for j, i in enumerate(parte):
            mapa[chaves[i]] = (textos[i], vet[j])
        _salvar(chave, mapa, caminho)
        if mostrar_progresso:
            print("  embeddings %s: %d/%d novos (%.0fs)" % (chave, min(ini + bloco, len(faltam)),
                                                           len(faltam), time.time() - t0), flush=True)
    return torch.stack([mapa[k][1] for k in chaves])


def _salvar(chave, mapa, caminho):
    ks = list(mapa)
    # o 8B (GGUF) nao tem "hf": usa o nome da chave
    torch.save({"modelo": MODELOS[chave].get("hf", chave), "chaves": ks, "textos": [mapa[k][0] for k in ks],
                "vetores": torch.stack([mapa[k][1] for k in ks])}, caminho + ".tmp")
    os.replace(caminho + ".tmp", caminho)
