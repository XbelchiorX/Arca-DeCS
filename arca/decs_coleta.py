# -*- coding: utf-8 -*-
"""Coleta do acervo do ARCA (opcao D4). SOMENTE LEITURA.

Le todos os itens pela busca do DSpace e guarda, de cada um, o que a
indexacao precisa: titulos, resumos, palavras-chave, DECS atual e o DECS
humano original ([DECS-ORIGINAL]), quando o script ja tiver gravado no item.

Incremental: mescla com a coleta anterior (itens novos entram, itens
existentes sao atualizados). Salva em dados/acervo_arca.json.
"""
import json
import os

from . import config
from . import decs_rastreabilidade as R

CAMPOS_TITULO = ("dc.title", "dc.title.alternative")
CAMPOS_RESUMO = ("dc.description.abstract", "dc.description.abstracten",
                 "dc.description.abstractes", "dc.description.abstractfr",
                 "dc.description.abstractit")
# dc.subject.mesh guarda termos MeSH em ingles: o indice lexico reconhece o
# nome ingles do descritor, entao eles viram sinal forte para o mesmo DeCS.
CAMPOS_PALAVRA_CHAVE = ("dc.subject.other", "dc.subject.en", "dc.subject.es",
                        "dc.subject.fr", "dc.subject.mesh")


# idioma implicito no nome do campo, quando o valor nao declara language
# (mesma regra do script de ODS: dc.description.abstracten -> en)
_SUFIXO_IDIOMA = {"en": "en", "es": "es", "fr": "fr", "it": "it", "mesh": "en"}


def _vals(md, campos):
    saida = []
    for c in campos:
        sufixo = c.rsplit(".", 1)[-1]
        sufixo = sufixo[len("abstract"):] if sufixo.startswith("abstract") else sufixo
        for x in md.get(c) or []:
            v = (x.get("value") or "").strip()
            if v:
                saida.append({"v": v, "l": x.get("language") or _SUFIXO_IDIOMA.get(sufixo, "")})
    return saida


def extrair(obj):
    """Objeto de item da API do DSpace -> registro do acervo."""
    md = obj.get("metadata") or {}
    return {
        "uuid": obj.get("uuid"),
        "handle": obj.get("handle"),
        "titulos": _vals(md, CAMPOS_TITULO)[:20],
        "resumos": _vals(md, CAMPOS_RESUMO)[:10],
        "keywords": _vals(md, CAMPOS_PALAVRA_CHAVE)[:100],
        "decs": [(x.get("value") or "").strip() for x in md.get(config.CAMPO_DECS) or []
                 if (x.get("value") or "").strip()],
        "decs_original": R.ler_original(md),
        # ODS: valores atuais e o que existia antes da 1a gravacao do script
        "ods": [(x.get("value") or "").strip() for x in md.get("dc.subject.ods") or []
                if (x.get("value") or "").strip()],
        "ods_original": R.ler_original(md, prefixo=R.PREFIXO_ODS_ORIGINAL),
        "lastModified": obj.get("lastModified", ""),
        "dc.date.accessioned": [{"value": x.get("value", ""), "l": ""}
                                for x in md.get("dc.date.accessioned") or []],
    }


def coletar(cliente, progresso=print, tamanho_pagina=100):
    """Percorre o acervo inteiro e atualiza dados/acervo_arca.json."""
    dados = {}
    if os.path.exists(config.ACERVO_JSON):
        with open(config.ACERVO_JSON, encoding="utf-8") as f:
            dados = json.load(f)
    antes = len(dados)
    url = "/discover/search/objects"
    params = {"query": "*:*", "dsoType": "ITEM", "size": tamanho_pagina, "page": 0}
    novos = atualizados = 0
    total_paginas = 1
    while params["page"] < total_paginas:
        j = cliente.get(url, params=params)
        sr = j.get("_embedded", {}).get("searchResult", {})
        total_paginas = int((sr.get("page") or {}).get("totalPages", 1) or 1)
        for o in (sr.get("_embedded") or {}).get("objects", []):
            obj = o.get("_embedded", {}).get("indexableObject") or {}
            if not obj.get("handle") or obj.get("type") != "item":
                continue          # colecao/comunidade nao e documento (1.562 entraram antes)
            reg = extrair(obj)
            if obj["handle"] in dados:
                atualizados += 1
            else:
                novos += 1
            dados[obj["handle"]] = reg
        params["page"] += 1
        if params["page"] % 20 == 0:
            progresso("  pagina %d/%d (%d itens)" % (params["page"], total_paginas, len(dados)))
        if params["page"] % 150 == 0:
            _salvar(dados)          # checkpoint (raro: o arquivo tem ~200 MB)
    _salvar(dados)
    return {"antes": antes, "depois": len(dados), "novos": novos, "atualizados": atualizados,
            "com_decs": sum(1 for x in dados.values() if x.get("decs"))}


def _salvar(dados):
    """Grava em .tmp e troca. O OneDrive trava o arquivo enquanto sincroniza
    (WinError 32 medido em 2026-10-02 com o acervo de 196 MB): tenta de novo
    por ate 2 minutos antes de desistir."""
    import time
    tmp = config.ACERVO_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False)
    for tentativa in range(60):
        try:
            os.replace(tmp, config.ACERVO_JSON)
            return
        except PermissionError:
            time.sleep(2)
    raise PermissionError("acervo_arca.json travado (OneDrive?) por mais de 2 minutos")
