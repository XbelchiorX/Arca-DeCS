# -*- coding: utf-8 -*-
"""So leitura: sorteia N itens gravados pelo sugerir_sem_decs.py (SQLite, aplicado=1, DECS-ORIGINAL vazio) desde uma
data e escreve, por item, titulo + resumo inteiro + palavras-chave + termos gravados (com metodo), para avaliar.
Uso: python ferramentas/decs/amostra_decs_auto.py 2026-10-20T21:00 100"""
import json
import os
import random
import sqlite3
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config, dspace  # noqa: E402

desde, n = sys.argv[1].replace("T", " "), int(sys.argv[2])
db = sqlite3.connect(config.BANCO_SQLITE)
rs = db.execute("select handle, decs_depois, mudancas_detalhadas from indexacao_log where aplicado=1 and data>=? "
                "and decs_antes='[]'", (desde,)).fetchall()
random.seed(20261020)
am = random.sample(rs, min(n, len(rs)))
cli = dspace.conectar()
saida = os.path.join(config.SAIDAS_DECS, "amostra_decs_auto_%d.md" % len(am))
with open(saida, "w", encoding="utf-8") as f:
    f.write("# Amostra de %d itens (de %d gravados)\n\n" % (len(am), len(rs)))
    for h, dep, mud in am:
        md = (cli.item_por_handle(h) or {}).get("metadata") or {}
        v = lambda c: [x["value"] for x in md.get(c) or []]
        f.write("## %s\nTITULO: %s\nRESUMO: %s\nPALAVRAS-CHAVE: %s\nGRAVADO:\n" % (
            h, " | ".join(v("dc.title")), " || ".join(v("dc.description.abstract")),
            "; ".join(v("dc.subject.other") + v("dc.subject.en"))))
        for l in json.loads(mud):
            f.write("  - [%s] %s\n" % (l.get("metodo", ""), l.get("depois", "")))
        f.write("\n")
print(saida)
