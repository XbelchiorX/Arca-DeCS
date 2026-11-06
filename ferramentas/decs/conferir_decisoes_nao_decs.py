# -*- coding: utf-8 -*-
"""Mostra termo original -> decisao, lado a lado, para conferir alinhamento. Uso: ... decisoes.txt [ini fim]"""
import json
import sys

cand = json.load(open("saidas/decs/nao_decs/candidatos.json", encoding="utf-8"))
ini, fim = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1, 10 ** 9)
for l in open(sys.argv[1], encoding="utf-8"):
    p = l.rstrip("\n").split("|")
    if not p[0].isdigit():
        continue
    i = int(p[0])
    if ini <= i <= fim and p[1] in ("D", "X"):
        print("%d %s | %s -> %s" % (i, p[1], cand[i - 1]["valor"].replace("\n", " ")[:60], p[2] if len(p) > 2 else ""))
