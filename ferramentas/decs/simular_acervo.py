# -*- coding: utf-8 -*-
"""Simula o plano de DECS para o acervo inteiro (o mesmo que a opcao D3), sem perguntas.

SOMENTE LEITURA: usa a copia local do acervo (dados/acervo_arca.json) e nao
acessa o ARCA. Gera saidas/decs/plano_todos_*.csv e um resumo em
saidas/decs/resumo_simulacao.md.

Uso:  python ferramentas/decs/simular_acervo.py [--amostra 5000] [--sem-sugestoes]
      (sem --amostra: acervo inteiro, ~40 min numa RTX 3060 Ti)
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from arca import config, decs_menu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-sugestoes", action="store_true")
    ap.add_argument("--amostra", type=int, default=0,
                    help="N itens sorteados (metade com DECS, metade sem); 0 = todos")
    args = ap.parse_args()
    _ctx, acervo_todo = decs_menu.contexto()
    handles = sorted(acervo_todo)
    if args.amostra:
        rnd = random.Random(20261002)
        com = [h for h in handles if acervo_todo[h].get("decs")]
        sem = [h for h in handles if not acervo_todo[h].get("decs")]
        handles = sorted(rnd.sample(com, args.amostra // 2) + rnd.sample(sem, args.amostra - args.amostra // 2))
    acervo = {h: acervo_todo[h] for h in handles}
    caminho, rel, status = decs_menu._processar(handles, "amostra" if args.amostra else "todos",
                                                sugerir=not args.sem_sugestoes, gravar=False)
    # resumo legivel
    por_item = defaultdict(Counter)
    with open(caminho, encoding="utf-8-sig") as f:
        for l in csv.DictReader(f, delimiter=";"):
            por_item[l["handle"]][l["acao"]] += 1
    com_decs = sum(1 for h in acervo if acervo[h].get("decs"))
    ganham = sum(1 for c in por_item.values() if c["ADICIONAR"])
    sem_ganham = sum(1 for h, c in por_item.items() if c["ADICIONAR"] and not acervo[h].get("decs"))
    linhas = ["# Simulacao do DECS (%s)" % ("amostra de %d itens" % len(acervo) if args.amostra
                                            else "acervo inteiro"), "",
              "Plano completo: `%s`" % os.path.relpath(caminho, config.RAIZ), "",
              "| | itens |", "|---|---:|",
              "| simulados | %d |" % len(acervo),
              "| com DECS do catalogador | %d |" % com_decs,
              "| que mudariam | %d |" % rel.itens_com_mudanca,
              "| que ganhariam descritor automatico | %d |" % ganham,
              "| ... dos quais nao tinham nenhum DECS | %d |" % sem_ganham, "",
              "| Acao | valores |", "|---|---:|"]
    for a, n in rel.contagem.most_common():
        linhas.append("| %s | %d |" % (a, n))
    with open(os.path.join(config.SAIDAS_DECS, "resumo_simulacao.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
