# -*- coding: utf-8 -*-
"""So leitura: o D2 e idempotente? Para itens que JA passaram pelo D1/D2, simula de novo exatamente o que o D2
faria (as 4 correcoes, sem --gravar, e as sugestoes com a opcao "todas") e mostra o que ainda mudaria.
Se o D2 for idempotente, a 2a passada nao muda nada. Pergunta do responsavel (2026-10-21).
Uso: python ferramentas/decs/testar_repeticao_d2.py icict/79038 icict/67821 ...   (ou --amostra N)
"""
import json
import os
import random
import sqlite3
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
os.environ["ARCA_SOMENTE_LEITURA"] = "sim"
from arca import config  # noqa: E402

SCRIPTS = (("corrigir_termos_fora_do_decs.py", []), ("corrigir_raiz_decs.py", []),
           ("galho_termos_soltos.py", ["--completar"]), ("retirar_sem_arvore_decs.py", []))


def amostra(n):
    db = sqlite3.connect(config.BANCO_SQLITE)
    hs = sorted({h for (h,) in db.execute("select handle from indexacao_log where aplicado=1 and data>='2026-10-19'")})
    random.seed(20261021)
    return random.sample(hs, min(n, len(hs)))


def main():
    a = sys.argv[1:]
    hs = amostra(int(a[1])) if a[:1] == ["--amostra"] else a
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    print("== 1) correcoes (simulacao) em %d itens" % len(hs), flush=True)
    for script, extra in SCRIPTS:
        r = subprocess.run([sys.executable, os.path.join(RAIZ, "ferramentas", "decs", script), "--item"] + hs + extra,
                           cwd=RAIZ, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        linhas = [l for l in r.stdout.splitlines() if l.strip()]
        mudaria = [l for l in linhas if ("a_corrigir" in l or "a_gravar" in l or "trocaria" in l)]
        fim = [l for l in linhas if l.startswith("FIM")]
        print("  %-34s %s" % (script, (fim[-1] if fim else "") + ("" if not mudaria else " | MUDARIA: " + "; ".join(mudaria[:5]))),
              flush=True)
    print("== 2) sugestoes (opcao 'todas')", flush=True)
    from arca import decs_correcao as C, decs_menu as M, dspace
    ctx, _ = M.contexto(sugerir=True)
    cli = dspace.conectar()
    total = 0
    for h in hs:
        obj = cli.item_por_handle(h)
        md = obj.get("metadata") or {}
        atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
        linhas, _f, _k = C.planejar(h, C.item_do_dspace(obj), ctx)
        tem = bool(atuais)
        novas = [l for l in linhas if l["acao"] == "ADICIONAR" or (not tem and l["acao"] == "SUGERIR")]
        # o mesmo filtro do gravar_novos: so entra o caminho que o item ainda nao tem
        entraria = [l["depois"] for l in novas if l.get("depois") and l["depois"] not in atuais]
        ja_tem_ui = [l["depois"] for l in novas if l.get("depois") and l["depois"] not in atuais
                     and any(v.split("::")[-1] == l["depois"].split("::")[-1] for v in atuais)]
        total += len(entraria)
        print("  %s: %d termos hoje -> entraria(m) %d %s%s" % (
            h, len(atuais), len(entraria), json.dumps([e.split("::")[-1] for e in entraria], ensure_ascii=False),
            ("  << MESMO DESCRITOR EM OUTRO GALHO: %s" % [j.split("::")[-1] for j in ja_tem_ui]) if ja_tem_ui else ""),
              flush=True)
    print("TOTAL de termos que uma 2a passada acrescentaria: %d" % total)


if __name__ == "__main__":
    main()
