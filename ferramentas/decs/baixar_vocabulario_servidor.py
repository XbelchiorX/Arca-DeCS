# -*- coding: utf-8 -*-
"""Baixa o vocabulario `decs` EXATAMENTE como o DSpace do ARCA o serve (opcao D6).

Por que existe: o valor gravado em dc.subject.decs so e reconhecido pelo
DSpace (arvore de navegacao, formulario de edicao) se for identico ao caminho
que o vocabulario do SERVIDOR produz. O XML local pode ser de outra versao do
DeCS (medido em 2026-10-02: 97% dos nos iguais, 1,7% renomeados, 1% ausentes).
Por isso a referencia de escrita e esta copia.

Como: percorre a arvore pela API (vocabularyEntryDetails .../children), nivel
a nivel. SOMENTE LEITURA.
  - Salva o progresso a cada 2.000 pedidos e RETOMA de onde parou se for
    interrompido (a 1a versao perdeu 45 mil nos ao ser cortada em 1 hora).
  - So pede os filhos de nos que tem filhos na arvore da BIREME: ~2/3 dos nos
    sao folhas e pedir os filhos delas era a maior parte do tempo. (Um no que
    so exista no servidor abaixo de uma folha da BIREME fica de fora; os
    valores antigos que apontarem para ele sao reconhecidos pelo nome.)

Saida: dados/vocabulario_servidor_decs.json
    {"baixado_em", "servidor", "nos": {"C01.069": {"valor", "rotulo", "pai", "ui"}}}

Uso:  python ferramentas/decs/baixar_vocabulario_servidor.py
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from datetime import datetime

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from arca import config, dspace  # noqa: E402
from arca import decs_vocabulario as V  # noqa: E402

SAIDA = config.VOCAB_SERVIDOR_JSON
PARCIAL = SAIDA + ".parcial"
# nota do vocabulario publicado hoje: "ID DeCS: D000785"; do v2.1: "ID: D000785"
_RE_UI = re.compile(r"\bID(?: DeCS)?:\s*(\S+)")


def _no(e, pai):
    nota = (e.get("otherInformation") or {}).get("note") or ""
    m = _RE_UI.search(nota)
    return {"valor": e["value"], "rotulo": e["display"], "pai": pai, "ui": m.group(1) if m else None}


def main():
    cliente = dspace.conectar()
    voc = config.VOCABULARIO_DSPACE
    tem_filhos = {p for tn in V.arvore_da_base(V.carregar_base()) for p in V.prefixos_da_arvore(tn)[:-1]}

    def pagina(caminho):
        itens, pg = [], 0
        while True:
            j = cliente.get(caminho, params={"page": pg, "size": 1000})
            emb = j.get("_embedded", {})
            itens += emb.get("children", emb.get("vocabularyEntryDetails", []))
            pg += 1
            if pg >= (j.get("page") or {}).get("totalPages", 1):
                return itens

    if os.path.exists(PARCIAL):
        with open(PARCIAL, encoding="utf-8") as f:
            est = json.load(f)
        nos, fila = est["nos"], est["fila"]
        print("Retomando: %d nos ja baixados, %d na fila." % (len(nos), len(fila)), flush=True)
    else:
        nos, fila = {}, []
        for e in pagina("/submission/vocabularyEntryDetails/search/top?vocabulary=%s" % voc):
            tn = e["id"].split(":", 1)[1]
            nos[tn] = _no(e, None)
            fila.append(tn)

    def salvar_parcial():
        with open(PARCIAL, "w", encoding="utf-8") as f:
            json.dump({"nos": nos, "fila": fila}, f, ensure_ascii=False)

    def filhos(tn):
        return tn, pagina("/submission/vocabularyEntryDetails/%s:%s/children" % (voc, tn))

    t0 = time.time()
    feitos = 0
    with cf.ThreadPoolExecutor(16) as ex:
        while fila:
            # categorias de topo (A, C, SP...) sempre tem filhos
            lote = [tn for tn in fila if tn in tem_filhos or re.fullmatch(r"[A-Z]{1,2}", tn)][:2000]
            resto = set(lote)
            fila = [tn for tn in fila if tn not in resto and (tn in tem_filhos or re.fullmatch(r"[A-Z]{1,2}", tn))]
            for pai, lista in ex.map(filhos, lote):
                for e in lista:
                    tn = e["id"].split(":", 1)[1]
                    if tn not in nos:
                        nos[tn] = _no(e, pai)
                        fila.append(tn)
            feitos += len(lote)
            salvar_parcial()
            print("  %d nos | %d pedidos | fila %d | %.0fs" % (len(nos), feitos, len(fila),
                                                             time.time() - t0), flush=True)

    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"baixado_em": datetime.now().isoformat(timespec="seconds"),
                   "servidor": config.BASE_URL, "nos": nos}, f, ensure_ascii=False)
    os.remove(PARCIAL)
    print("OK: %d nos -> %s" % (len(nos), SAIDA))


if __name__ == "__main__":
    main()
