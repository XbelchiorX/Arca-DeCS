# -*- coding: utf-8 -*-
"""So leitura: mostra o plano do DeCS (planejar) para os itens dados. Uso: python ferramentas/decs/teste_plano_item.py icict/67821"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arca import decs_correcao as C, decs_menu as M, dspace  # noqa: E402

ctx, _ = M.contexto(sugerir=True)
cli = dspace.conectar()
for h in sys.argv[1:]:
    obj = cli.item_por_handle(h)
    linhas, _f, _k = C.planejar(h, C.item_do_dspace(obj), ctx)
    print("==", h)
    for l in linhas:
        print("  %-11s %-20s %s %s" % (l["acao"], l["metodo"], l["depois"] or l["antes"], l["probabilidade"]))
