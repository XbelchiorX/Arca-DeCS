# -*- coding: utf-8 -*-
"""Planilha para o responsavel APROVAR o galho que o modelo escolhe onde o catalogador nao gravou caminho.

Decisao do responsavel (2026-10-17): a referencia do galho e o que os catalogadores ja gravaram no sistema. O modelo
de galho aprende com os ~20% de valores que tem caminho; esta planilha confere se ele generaliza para os ~80% que sao
termos soltos (o medo do responsavel). Casos: itens do teste (500 ultimos depositados com DeCS) em que o catalogador
pos um termo SOLTO de descritor com mais de um caminho; o galho e o do modelo do Qwen3-8B, sem o item no treino.
No maximo 2 casos por descritor, sorteio com semente fixa.
Saida: saidas/avaliacao/aprovacao_galho_150.xlsx
Uso: python ferramentas/decs/planilha_galho_aprovacao.py [--n 150]
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "ferramentas", "decs"))
os.environ.setdefault("DECS_MODELO_EMBEDDING", "qwen3-8b")
import torch  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from arca import config, decs_acervo, decs_galho, decs_indexador as IX, decs_vocabulario as V  # noqa: E402
from arca.decs_indexador import valores_humanos  # noqa: E402
from arca.decs_interpretar import NOME, SINONIMO, interpretar  # noqa: E402
from treinar_avaliar import selecionar_amostra  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    acervo, ordem = selecionar_amostra(decs_acervo.carregar_acervo(), vocab, 500, 4000, 8000)
    teste = ordem[:500]
    decs_galho.ARQ_MODELO_GALHO = os.path.join(config.DADOS_LOCAIS, "modelos_qwen3-8b", "modelo_galho.pkl")
    c = torch.load(IX.ARQ_EMB_ITENS, weights_only=False)
    pos = {k: i for i, k in enumerate(c["chaves"])}
    handles = [h for h in acervo if h in pos]
    vetores = c["vetores"][[pos[h] for h in handles]]
    galho = decs_galho.EscolhaDeGalho(vocab, acervo, vetores=vetores, handles=handles)
    if galho.modelo is None:
        sys.exit("modelo de galho do 8B nao encontrado")
    vet = {h: vetores[i] for i, h in enumerate(handles)}

    casos = []
    for h in teste:
        vals = valores_humanos(acervo[h])
        for v in vals:
            r = interpretar(v, vocab)
            if r["tipo"] in (NOME, SINONIMO) and r["ui"] and len(vocab.caminhos(r["ui"])) > 1:
                casos.append((h, r["ui"], v, vals))
    random.Random(20261017).shuffle(casos)
    por_desc, escolhidos = Counter(), []
    for caso in casos:
        if por_desc[caso[1]] < 2 and len(escolhidos) < args.n:
            por_desc[caso[1]] += 1
            escolhidos.append(caso)

    wb = Workbook()
    ws = wb.active
    ws.title = "Aprovar galho"
    cab = ["#", "Item", "Link", "Titulo", "Termo do catalogador (solto)", "Descritor", "Galho do modelo",
           "Uso no acervo (quantas vezes cada galho foi gravado)", "Todos os galhos possiveis",
           "APROVA? (S/N)", "Se N: numero do galho certo", "Observacao"]
    ws.append(cab)
    saida_json = []
    for i, (h, u, v, vals) in enumerate(escolhidos, 1):
        ctx = galho.contexto(h, vals, vet.get(h))
        cam, _m = galho.escolher(u, excluir_handle=h, contexto=ctx)
        cams = vocab.caminhos(u)
        uso = galho.uso.get(u, Counter())
        uso_txt = "\n".join("%d. %d" % (k, uso.get(cm, 0)) for k, cm in enumerate(cams, 1))
        todos = "\n".join("%d. %s" % (k, cm) for k, cm in enumerate(cams, 1))
        tit = (decs_acervo._valores(acervo[h], "titulos") or [""])[0]
        ws.append([i, h, "https://arca.fiocruz.br/handle/" + h, tit, v, vocab.nome(u),
                   "%d. %s" % (cams.index(cam) + 1, cam), uso_txt, todos, "", "", ""])
        saida_json.append({"n": i, "item": h, "ui": u, "termo": v, "galho_modelo": cam, "galhos": cams})
    larg = [5, 13, 18, 45, 28, 28, 60, 22, 80, 12, 14, 30]
    for k, w in enumerate(larg):
        ws.column_dimensions[chr(65 + k)].width = w
    for cel in ws[1]:
        cel.font = Font(bold=True, color="FFFFFF")
        cel.fill = PatternFill("solid", fgColor="1F4E78")
        cel.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for cel in row:
            cel.alignment = Alignment(wrap_text=True, vertical="top")
        row[9].fill = PatternFill("solid", fgColor="FFF2CC")
        row[10].fill = PatternFill("solid", fgColor="FFF2CC")
    dv = DataValidation(type="list", formula1='"S,N"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add("J2:J%d" % (len(escolhidos) + 1))
    ws.freeze_panes = "E2"
    ws2 = wb.create_sheet("Como preencher")
    for linha in ["Referencia do galho: o que os catalogadores ja gravaram no ARCA (decisao do responsavel, 2026-10-17).",
                  "Cada linha: o catalogador pos o termo SOLTO (sem caminho); o modelo escolheu o galho da coluna G.",
                  "Coluna J: S se o galho do modelo esta certo para ESTE item; N se nao.",
                  "Coluna K (so quando N): o numero do galho certo, da lista da coluna I.",
                  "A coluna H mostra quantas vezes cada galho ja foi gravado pelos catalogadores (o costume da casa).",
                  "Depois de preenchida, o resultado mede se o modelo generaliza para os termos soltos do acervo."]:
        ws2.append([linha])
    ws2.column_dimensions["A"].width = 120
    out = os.path.join(config.SAIDAS_AVALIACAO, "aprovacao_galho_150.xlsx")
    wb.save(out)
    json.dump(saida_json, open(out.replace(".xlsx", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("planilha: %s (%d casos de %d termos soltos multigalho; %d descritores)"
          % (out, len(escolhidos), len(casos), len(por_desc)))


if __name__ == "__main__":
    main()
