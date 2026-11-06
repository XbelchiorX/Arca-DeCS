# -*- coding: utf-8 -*-
"""So leitura. Regra 8 para a regra da palavra-chave do autor (DECISOES §18): nos itens com DeCS HUMANO, quantas
vezes a palavra-chave que e nome/sinonimo exato de um descritor coincide com o DeCS do catalogador. Por palavra-chave
(normalizada): n e acerto. Os sinonimos de acerto baixo sao candidatos a "traicoeiros".
Saida: saidas/decs/sinonimos/medicao.json e .md"""
import json
import os
import sys
from collections import defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config, decs_acervo as A, decs_interpretar as I, decs_vocabulario as V  # noqa: E402
from arca.decs_indexador import descritores_humanos  # noqa: E402
from arca.decs_vocabulario import norm  # noqa: E402

v = V.carregar_vocabulario()
a = A.carregar_acervo()
cache, st = {}, defaultdict(lambda: [0, 0, "", "", ""])
tot = {I.NOME: [0, 0], I.SINONIMO: [0, 0]}
for h, r in a.items():
    g = set(descritores_humanos(r, v))
    if not g:
        continue
    for k in r.get("keywords") or []:
        t = k.get("v", "")
        if t not in cache:
            x = I.interpretar(t, v)
            cache[t] = (x["tipo"], x["ui"]) if x["tipo"] in (I.NOME, I.SINONIMO) and x.get("ui") else None
        c = cache[t]
        if not c:
            continue
        ok = c[1] in g
        tot[c[0]][0] += ok
        tot[c[0]][1] += 1
        s = st[norm(t)]
        s[0] += ok
        s[1] += 1
        s[2], s[3], s[4] = t, c[0], v.nome(c[1])
linhas = sorted(({"termo": s[2], "tipo": s[3], "descritor": s[4], "acertos": s[0], "n": s[1],
                  "acerto": round(s[0] / s[1], 3)} for s in st.values()), key=lambda d: (d["acerto"], -d["n"]))
pasta = os.path.join(config.SAIDAS_DECS, "sinonimos")
json.dump({"total": {k: {"acertos": x[0], "n": x[1], "acerto": round(x[0] / max(1, x[1]), 3)} for k, x in tot.items()},
           "termos": linhas}, open(os.path.join(pasta, "medicao.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(pasta, "medicao.md"), "w", encoding="utf-8") as f:
    for k, x in tot.items():
        f.write("%s: %d/%d = %.1f%%\n" % (k, x[0], x[1], 100 * x[0] / max(1, x[1])))
    f.write("\nSinonimos com n>=3 e acerto < 30%:\n")
    for d in linhas:
        if d["tipo"] == I.SINONIMO and d["n"] >= 3 and d["acerto"] < 0.3:
            f.write("- %s -> %s (%d/%d)\n" % (d["termo"], d["descritor"], d["acertos"], d["n"]))
print("ok", {k: x for k, x in tot.items()}, flush=True)
