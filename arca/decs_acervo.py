# -*- coding: utf-8 -*-
"""O acervo coletado do ARCA (opcao D4) e o texto de cada item.

Formato de dados/acervo_arca.json (um registro por handle):
    {"icict/12345": {"uuid": ..., "titulos": [{"v": texto, "l": idioma}],
                     "resumos": [...], "keywords": [...], "decs": [valores],
                     "lastModified": ..., "dc.date.accessioned": [...]}}
"""
import json
import os

from . import config


def carregar_acervo(caminho=None):
    caminho = caminho or config.ACERVO_JSON
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            "Acervo nao coletado (%s). Rode a opcao D4 do menu." % caminho)
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    for h, r in dados.items():      # 179 registros de coleta antiga nao guardam o handle no registro
        r.setdefault("handle", h)
    return dados


def _valores(item, campo):
    return [(x.get("v") or "").strip() for x in item.get(campo, []) if (x.get("v") or "").strip()]


def texto_do_item(item, max_resumo=2500):
    """Texto que representa o documento.

    Ordem: titulo, palavras-chave, resumo. O modelo trunca em 512 tokens, e o
    que vem primeiro e o que sobrevive; palavras-chave sao o campo mais denso
    em assunto (medido na v8.0.2: com o resumo antes, um item de 900 palavras
    perdia as 6 palavras-chave). Todos os idiomas entram: o modelo e
    multilingue, e um titulo em ingles + resumo em portugues reforcam o mesmo
    assunto.

    Os DECS do proprio item NUNCA entram: sao a resposta, nao a pergunta.
    """
    titulos = " ".join(_valores(item, "titulos"))
    kws = "; ".join(_valores(item, "keywords"))
    resumo = " ".join(_valores(item, "resumos"))
    if max_resumo:
        resumo = resumo[:max_resumo]
    partes = [titulos]
    if kws:
        partes.append("Palavras-chave: " + kws)
    if resumo:
        partes.append(resumo)
    return "\n".join(p for p in partes if p)


def tem_texto_suficiente(item):
    """Item com titulo e (resumo ou palavras-chave) — o minimo para indexar."""
    return bool(_valores(item, "titulos")) and bool(
        _valores(item, "resumos") or _valores(item, "keywords"))
