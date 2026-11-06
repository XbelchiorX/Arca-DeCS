# -*- coding: utf-8 -*-
"""Confere as decisoes do Claude para termos fora do DeCS: todo 'D|nome' tem de apontar a UM descritor gravavel.
Uso: python ferramentas/decs/validar_decisoes_nao_decs.py saidas/decs/nao_decs/decisoes_*.txt
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import decs_vocabulario as V  # noqa: E402


def resolver(v, t):
    if t in v.base:
        return [t]
    return [u for u in v.uis_do_termo(t)[0] if v.caminhos(u)]


def main():
    v = V.carregar_vocabulario()
    ruins = 0
    for arq in sys.argv[1:]:
        for l in open(arq, encoding="utf-8"):
            p = l.strip().split("|")
            if len(p) >= 3 and p[1] == "D":
                uis = resolver(v, p[2])
                if len(uis) == 1 and p[2] not in v.base and v.uis_do_termo(p[2])[1] != "nome":
                    print("  sinonimo:", p[0], p[2], "->", v.nome(uis[0]))
                if len(uis) != 1:
                    ruins += 1
                    print(p[0], p[2], "->", [(u, v.nome(u)) for u in uis])
    print("problemas:", ruins)


if __name__ == "__main__":
    main()
