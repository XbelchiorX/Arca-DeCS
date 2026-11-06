# -*- coding: utf-8 -*-
"""Busca meta a meta: o documento contra as 180 metas do Ipea (2026), "Nacionalizacao da
Agenda 2030" (169 globais + 14.8 brasileira + ODS 18).

Modelo (decisao D17): Qwen3-Embedding-8B (2o de 93 no MTEB-BR 2026, empatado com o 1o; 4o no MTEB
Multilingual v2; o acervo e ~37% so em ingles), GGUF oficial Q8_0 rodando no llama-server (CUDA,
camadas que nao cabem nos 8 GB ficam na RAM). O servidor recusa texto maior que o lote em vez de
cortar. ODS_EMBEDDING=Qwen/Qwen3-Embedding-0.6B volta ao modelo anterior (sentence-transformers). Cada meta e
representada pelo texto global (ONU) E pelo texto Brasil (decisao do responsavel, 03/10/2026),
mais os conceitos importantes do Ipea. Tudo inteiro. O documento entra inteiro (titulo + resumo
+ palavras-chave). Saidas: similaridade por meta e, por ODS, a maior delas.
"""
import json
import os

import numpy as np

from . import config

MODELO = os.getenv("ODS_EMBEDDING", "Qwen/Qwen3-Embedding-8B-GGUF")
ARQ_GGUF = os.getenv("ODS_EMBEDDING_ARQ", "Qwen3-Embedding-8B-Q8_0.gguf")
PORTA_EMB = 8092
N_CANDIDATAS = int(os.getenv("ODS_N_METAS", "20"))   # metas entregues ao professor (D18: cobertura 95,5% dos ODS sem o 3 no gabarito); ODS_N_METAS so para experimento
LOTE_TOKENS = 16384          # maior texto aceito de uma vez (o servidor recusa o que passar; nada e cortado)
LOTE_CHARS = 6000             # textos curtos juntos num pedido ate esta soma de caracteres (bem abaixo dos 16k tokens)
METAS_JSON = os.path.join(config.DADOS, "ods_referencia", "metas_2026.json")   # ferramentas/ods/referencia/extrair_metas_2026.py
INSTRUCAO = ("Instruct: Dado o titulo, o resumo e as palavras-chave de um documento cientifico, "
             "encontre as metas dos Objetivos de Desenvolvimento Sustentavel que ele aborda\nQuery: ")
_M = {}


def metas():
    if "metas" not in _M:
        with open(METAS_JSON, encoding="utf-8") as f:
            _M["metas"] = json.load(f)
    return _M["metas"]


def texto_metas(m):
    """Texto global e texto Brasil, inteiros (um so quando o Ipea manteve a meta global)."""
    onu, br = m.get("onu") or "", m.get("brasil") or ""
    if not onu or m.get("mantida") or onu == br:
        return br or onu
    return "Global (ONU): %s | Brasil (Ipea 2026): %s" % (onu, br)


def texto_meta(cod, m):
    s = "ODS %d, meta %s: %s" % (m["ods"], cod, texto_metas(m))
    if m.get("conceitos"):
        s += " Conceitos: " + m["conceitos"]
    return s


def _gguf():
    return "GGUF" in MODELO


def _servidor():
    """Sobe o llama-server de embedding (uma vez por processo) e devolve a URL."""
    url = "http://127.0.0.1:%d" % PORTA_EMB
    if "proc" in _M and _M["proc"].poll() is None:
        return url
    import subprocess
    import time
    import requests
    # ja ha um servidor de embedding nesta porta (de outro processo): usa-o em vez de subir um segundo modelo,
    # que disputaria a GPU com o primeiro (aconteceu em 2026-10-19: 4x mais lento)
    if _M.get("externo"):
        return url
    try:
        if requests.get(url + "/health", timeout=3).status_code == 200:
            _M["externo"] = True
            return url
    except Exception:
        pass
    from huggingface_hub import hf_hub_download
    from . import ods_professor as P
    gguf = hf_hub_download(MODELO, ARQ_GGUF)
    amb = dict(os.environ)
    try:
        import torch
        amb["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + amb.get("PATH", "")
    except Exception:
        pass
    log = open(os.path.join(config.LOGS, "llama_server_embedding.log"), "w")
    _M["proc"] = subprocess.Popen(
        [P.LLAMA_SERVER, "-m", gguf, "--embeddings", "--pooling", "last"]
        + (["-ngl", os.getenv("ODS_EMB_NGL")] if os.getenv("ODS_EMB_NGL") else [])   # sem -ngl: o llama.cpp decide o que cabe na GPU
        + ["-c", str(LOTE_TOKENS), "-b", str(LOTE_TOKENS), "-ub", str(LOTE_TOKENS), "-np", "1", "-fa", "on",
         "--host", "127.0.0.1", "--port", str(PORTA_EMB)], env=amb, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(200):
        if _M["proc"].poll() is not None:
            raise RuntimeError("servidor de embedding caiu (veja logs/llama_server_embedding.log)")
        try:
            if requests.get(url + "/health", timeout=3).status_code == 200:
                return url
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("servidor de embedding nao subiu (veja logs/llama_server_embedding.log)")


def liberar():
    """Libera a GPU (modelo do sentence-transformers e/ou servidor de embedding)."""
    _M.pop("modelo", None)
    _M.pop("externo", None)   # servidor de outro processo: nao e nosso para encerrar
    p = _M.pop("proc", None)
    if p is not None and p.poll() is None:
        p.terminate()
        p.wait(60)
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _encode(textos, lote=8):
    """Vetores normalizados. GGUF: o Qwen3-Embedding usa o ultimo token (<|endoftext|>), que o proprio
    llama.cpp acrescenta (metadado do modelo); acrescentar de novo gera EOS duplo e muda o vetor."""
    if not _gguf():
        return np.asarray(_modelo().encode(textos, normalize_embeddings=True, batch_size=lote,
                                           show_progress_bar=False), dtype=np.float32)
    import requests
    url = _servidor()

    def um_pedido(conteudo):
        r = requests.post(url + "/embedding", json={"content": conteudo}, timeout=600)
        if r.status_code != 200:
            raise RuntimeError("embedding recusado (%s): %s" % (r.status_code, r.text[:300]))
        vs = [np.asarray(x["embedding"], dtype=np.float32).reshape(-1) for x in r.json()]
        return [v / max(np.linalg.norm(v), 1e-12) for v in vs]

    # textos curtos vao juntos num pedido (medido em 2026-10-19: 8 textos curtos 4x mais rapido, cosseno >= 0,9998
    # com o pedido individual); texto longo vai sozinho, como antes
    out, lote_atual, chars = [], [], 0
    for t in textos:
        if lote_atual and (chars + len(t) > LOTE_CHARS or len(lote_atual) >= 32):
            out += um_pedido(lote_atual) if len(lote_atual) > 1 else um_pedido(lote_atual[0])
            lote_atual, chars = [], 0
        lote_atual.append(t)
        chars += len(t)
    if lote_atual:
        out += um_pedido(lote_atual) if len(lote_atual) > 1 else um_pedido(lote_atual[0])
    return np.stack(out)


def _modelo():
    if "modelo" not in _M:
        import torch
        from sentence_transformers import SentenceTransformer
        _M["modelo"] = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu",
                                           model_kwargs={"torch_dtype": torch.float16})
        _M["modelo"].max_seq_length = 8192      # cabe o maior documento do acervo inteiro
    return _M["modelo"]


def vetores_metas():
    if "vm" not in _M:
        ms = metas()
        _M["cods"] = list(ms)
        _M["vm"] = _encode([texto_meta(c, ms[c]) for c in _M["cods"]], lote=16)
    return _M["cods"], _M["vm"]


def pontuar(textos, lote=8):
    """Matriz (n_textos x n_metas) de similaridade cosseno."""
    cods, vm = vetores_metas()
    _M["ultimo_vd"] = _encode([INSTRUCAO + t for t in textos], lote)
    return cods, _M["ultimo_vd"] @ vm.T


def ultimo_vetor_documentos():
    """Vetores dos documentos da ultima chamada de pontuar() (para reaproveitar)."""
    return _M["ultimo_vd"]


def por_ods(cods, sim):
    """Maior similaridade por ODS (colunas 1..18)."""
    ms = metas()
    out = np.zeros((sim.shape[0], 18), dtype=np.float32) - 1
    for j, c in enumerate(cods):
        o = ms[c]["ods"] - 1
        out[:, o] = np.maximum(out[:, o], sim[:, j])
    return out


def candidatas(cods, linha, k=N_CANDIDATAS):
    """As k metas mais proximas (codigo, similaridade)."""
    idx = np.argsort(-linha)[:k]
    return [(cods[i], float(linha[i])) for i in idx]
