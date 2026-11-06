# -*- coding: utf-8 -*-
"""Retira os termos do CLASSIFICADOR (ou, com --traicoeiros, os de sinonimos traicoeiros: arca/decs_sinonimos_traicoeiros.py) gravados pela gravacao em massa de 2026-10-20 (ferramentas/decs/sugerir_sem_decs.py),
decisao do responsavel depois da amostra de 100 (Politica de Saude em 84% dos termos do modelo; "Eugenia" p=0,999 sem
relacao). Os termos das palavras-chave do autor ficam. So as janelas das gravacoes em massa (UTC, SQLite): nao toca
nos testes do D1/D2. Le cada item AO VIVO, retira so o valor exato gravado pelo modelo, confere depois e registra.
Se o item fica sem DeCS, sai tambem a linha [DECS-AUTO] ([DECS-ORIGINAL] vazio fica: diz que nao havia humano).
Uso: python ferramentas/decs/retirar_classificador_lote_20261020.py [--gravar]"""
import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402

TRAICO = "--traicoeiros" in sys.argv
JANELAS = [("2026-10-20 20:41", "2026-10-20 20:44"), ("2026-10-20 21:25", "2026-10-20 22:08")]


def _traicoeiro(l):
    import re
    from arca.decs_sinonimos_traicoeiros import traicoeiro
    m = re.match(r"palavra-chave do autor '(.*)' e descritor DeCS", l.get("motivo") or "")
    return bool(m and traicoeiro(m.group(1)))


def alvos(traicoeiros=False):
    db = sqlite3.connect(config.BANCO_SQLITE)
    por = defaultdict(set)
    for a, b in JANELAS:
        for h, mud in db.execute("select handle, mudancas_detalhadas from indexacao_log where aplicado=1 and "
                                 "decs_antes='[]' and data>=? and data<?", (a, b)):
            for l in json.loads(mud):
                if l.get("depois") and (_traicoeiro(l) if traicoeiros else l.get("metodo") == "classificador"):
                    por[h].add(l["depois"])
    return por


def um(cli, h, tirar, gravar):
    obj = cli.item_por_handle(h)
    md = obj.get("metadata") or {}
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    idx = [i for i, v in enumerate(atuais) if v in tirar]
    if not idx:
        return "ja_sem"
    if not gravar:
        return "a_retirar"
    ops = [{"op": "remove", "path": "/metadata/%s/%d" % (config.CAMPO_DECS, i)} for i in sorted(idx, reverse=True)]
    esperado = [v for i, v in enumerate(atuais) if i not in idx]
    if not esperado:
        ops.extend(R.ops_remover_linhas(md, (R.PREFIXO_AUTO,)))
    cli.patch("/core/items/" + obj["uuid"], ops)
    depois = [x.get("value", "") for x in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    R.registrar(h, atuais, depois, [{"acao": "RETIRAR", "antes": atuais[i], "depois": "",
                                     "motivo": "sinonimo traicoeiro" if TRAICO else "classificador desligado (amostra 2026-10-20)"} for i in idx],
                aplicado=True)
    return "retirado" if depois == esperado else "DIVERGENTE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravar", action="store_true")
    ap.add_argument("--traicoeiros", action="store_true",
                    help="em vez do classificador: os termos de palavras-chave da lista de sinonimos traicoeiros")
    a = ap.parse_args()
    from arca import idiomas as I
    por = alvos(a.traicoeiros)
    print("%d itens, %d termos (%s)" % (len(por), sum(map(len, por.values())),
                                        "sinonimos traicoeiros" if a.traicoeiros else "classificador"), flush=True)
    cont = Counter()
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(lambda h=h, t=t: um(I.get_thread_client()._c, h, t, a.gravar)): h for h, t in por.items()}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                cont[f.result()] += 1
            except Exception as e:
                cont["erro"] += 1
                print("  ERRO %s: %s" % (futs[f], str(e)[:200]), flush=True)
            if k % 100 == 0:
                print("%d/%d %s" % (k, len(futs), dict(cont)), flush=True)
    print("FIM %s" % dict(cont))


if __name__ == "__main__":
    main()
