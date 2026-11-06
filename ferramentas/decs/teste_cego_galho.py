# -*- coding: utf-8 -*-
"""Teste CEGO de galho (pedido do responsavel, 2026-10-18): "confio mais no que um modelo da do que na regra da casa".
Tres metodos para o galho de termos SOLTOS com varios galhos possiveis:
  A. regra da casa: o galho mais usado pelos catalogadores para o descritor;
  B. modelo de galho treinado nas escolhas dos catalogadores (vetores do Qwen3-Embedding-8B);
  C. conteudo puro: o 8B compara o texto do documento com o texto de cada galho (nada do costume da casa).
So entram casos em que os metodos discordam. A planilha mostra as opcoes EMBARALHADAS, sem dizer o metodo; o
responsavel escolhe. A chave fica em teste_cego_galho_chave.json (abrir so depois).
Uso: python ferramentas/decs/teste_cego_galho.py [--n 100]   |   python ferramentas/decs/teste_cego_galho.py --resultado
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
os.environ["DECS_MODELO_EMBEDDING"] = "qwen3-8b"
from arca import config  # noqa: E402

PASTA = os.path.join(config.SAIDAS_DECS, "galhos")
CHAVE = os.path.join(PASTA, "teste_cego_galho_chave.json")
PLAN = os.path.join(PASTA, "teste_cego_galho.xlsx")


def montar(n):
    import torch
    import galho_termos_soltos as GT
    from arca import decs_acervo, decs_embeddings
    from arca.decs_interpretar import interpretar
    GT.ARQ_MODELO = os.path.join(config.DADOS_LOCAIS, "modelo_galho_soltos_qwen3-8b.pkl")
    G = GT.Galhos()
    v = G.vocab
    acervo = decs_acervo.carregar_acervo()
    ja = {l.strip().split("/")[-1] for l in open(os.path.join(PASTA, "amostra_100.txt"), encoding="utf-8")}
    uuids_ja = ja
    casos = []
    for h, it in acervo.items():
        if it.get("uuid") in uuids_ja or h not in G.vet:
            continue
        for val in it.get("decs", []):
            if "::" in val:
                continue
            r = interpretar(val, v)
            if r["tipo"] in GT.TIPOS_TERMO and r.get("ui") and len(v.caminhos(r["ui"])) > 1:
                casos.append((h, val, r["ui"]))
    random.Random(20261018).shuffle(casos)
    casos = casos[:n * 4]
    # textos dos galhos (so os necessarios) para o metodo C
    cams = sorted({c for _h, _v, u in casos for c in v.caminhos(u)})
    vc = decs_embeddings.embutir([c.replace("::", " > ") for c in cams]).float()
    vc = torch.nn.functional.normalize(vc, dim=1)
    vcam = {c: vc[i] for i, c in enumerate(cams)}
    decs_embeddings.liberar()
    linhas, vistos = [], set()
    for h, val, u in casos:
        if len(linhas) >= n or (h, u) in vistos:
            continue
        it = acervo[h]
        ctx = G.g.contexto(h, it.get("decs", []), G.vet[h])
        b, conf = G.escolher(h, u, ctx)
        a = G.precedente(u)
        c = max(v.caminhos(u), key=lambda cm: float(vcam[cm] @ G.vet[h]))
        if len({a, b, c}) == 1:
            continue
        vistos.add((h, u))
        opcoes = list(dict.fromkeys([a, b, c]))
        random.Random(len(linhas)).shuffle(opcoes)
        linhas.append({"n": len(linhas) + 1, "handle": h, "uuid": it.get("uuid"), "termo": val, "ui": u,
                       "titulo": " ".join(decs_acervo._valores(it, "titulos"))[:300],
                       "palavras_chave": "; ".join(decs_acervo._valores(it, "keywords"))[:300],
                       "opcoes": opcoes, "metodos": {"A": a, "B": b, "C": c}, "confianca_B": round(conf, 2)})
    json.dump(linhas, open(CHAVE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "Teste cego"
    ws.append(["#", "Registro", "Título", "Palavras-chave do autor", "Termo DeCS", "Opção 1", "Opção 2", "Opção 3",
               "MELHOR (1, 2, 3 ou T = tanto faz)", "Observação"])
    for l in linhas:
        op = l["opcoes"] + [""] * (3 - len(l["opcoes"]))
        ws.append([l["n"], "https://arca.fiocruz.br/items/%s" % l["uuid"], l["titulo"], l["palavras_chave"],
                   v.nome(l["ui"])] + [o.replace("::", " > ") for o in op] + ["", ""])
    for k, w in zip("ABCDEFGHIJ", [5, 30, 50, 40, 26, 55, 55, 55, 16, 30]):
        ws.column_dimensions[k].width = w
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78")
        c.alignment = Alignment(wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        row[8].fill = PatternFill("solid", fgColor="FFF2CC")
    dv = DataValidation(type="list", formula1='"1,2,3,T"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add("I2:I%d" % (len(linhas) + 1))
    ws.freeze_panes = "F2"
    ws2 = wb.create_sheet("Como preencher")
    for t in ["Cada linha: um termo DeCS solto de um registro real e os galhos possíveis propostos por métodos diferentes.",
              "As opções estão embaralhadas e não dizem de qual método vieram (teste cego).",
              "Coluna I: o número da opção que você acha mais adequada para ESTE documento; T se tanto faz.",
              "Quando houver só 2 opções, dois métodos concordaram.",
              "Ao terminar, salve e me avise: eu revelo qual método você escolheu mais."]:
        ws2.append([t])
    ws2.column_dimensions["A"].width = 120
    wb.save(PLAN)
    print("planilha: %s (%d casos; %d candidatos examinados)" % (PLAN, len(linhas), len(casos)))
    print("concordancias:", Counter("AB" if l["metodos"]["A"] == l["metodos"]["B"] else
                                     "AC" if l["metodos"]["A"] == l["metodos"]["C"] else
                                     "BC" if l["metodos"]["B"] == l["metodos"]["C"] else "todos_diferentes"
                                     for l in linhas))


def resultado():
    import re
    import shutil
    import tempfile
    from openpyxl import load_workbook
    chave = {l["n"]: l for l in json.load(open(CHAVE, encoding="utf-8"))}
    # a planilha pode estar aberta no Excel: le uma copia
    copia = os.path.join(tempfile.gettempdir(), "teste_cego_galho_copia.xlsx")
    try:
        shutil.copyfile(PLAN, copia)
    except PermissionError:  # o Excel trava a leitura comum; o Copy-Item do Windows consegue
        import subprocess
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Copy-Item -LiteralPath '%s' -Destination '%s' -Force" % (PLAN, copia)], check=True)
    ws = load_workbook(copia)["Teste cego"]
    jul = os.path.join(PASTA, "teste_cego_julgamento_claude.txt")
    claude = dict(tuple(map(int, l.split(":"))) for l in open(jul, encoding="utf-8") if l.strip()) \
        if os.path.exists(jul) else {}
    votos, tf, n, conc, comp = Counter(), 0, 0, 0, 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[8] in (None, ""):
            continue
        n += 1
        esc = str(row[8]).strip().upper()
        if esc == "T":
            tf += 1
            continue
        l = chave[int(row[0])]
        # "1.2", "1,2" ou "1;2": mais de uma opcao serve
        galhos = {l["opcoes"][int(k) - 1] for k in re.findall(r"[123]", esc) if int(k) <= len(l["opcoes"])}
        for m, g in l["metodos"].items():
            votos[m] += g in galhos
        if l["n"] in claude:
            comp += 1
            conc += l["opcoes"][claude[l["n"]] - 1] in galhos
    print("respondidos: %d | tanto faz: %d | escolha do Claude aceita pelo responsavel: %d de %d" % (n, tf, conc, comp))
    for m, nome in (("A", "regra da casa"), ("B", "modelo treinado nos catalogadores"), ("C", "conteudo puro (8B)")):
        print("  %s %-35s escolhido em %d de %d (%.0f%%)" % (m, nome, votos[m], n - tf, 100 * votos[m] / max(1, n - tf)))


def respostas_humanas():
    """{n: conjunto de galhos aceitos pelo responsavel} (planilha) e {n: galho do Claude} (julgamento cego)."""
    import re
    import shutil
    import subprocess
    import tempfile
    from openpyxl import load_workbook
    chave = {l["n"]: l for l in json.load(open(CHAVE, encoding="utf-8"))}
    copia = os.path.join(tempfile.gettempdir(), "teste_cego_galho_copia.xlsx")
    try:
        shutil.copyfile(PLAN, copia)
    except PermissionError:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Copy-Item -LiteralPath '%s' -Destination '%s' -Force" % (PLAN, copia)], check=True)
    resp = {}
    for row in load_workbook(copia)["Teste cego"].iter_rows(min_row=2, values_only=True):
        if row[0] is None or row[8] in (None, ""):
            continue
        l = chave[int(row[0])]
        esc = str(row[8]).strip().upper()
        resp[l["n"]] = set(l["opcoes"]) if esc == "T" else \
            {l["opcoes"][int(k) - 1] for k in re.findall(r"[123]", esc) if int(k) <= len(l["opcoes"])}
    jul = os.path.join(PASTA, "teste_cego_julgamento_claude.txt")
    claude = {int(a): chave[int(a)]["opcoes"][int(b) - 1] for a, b in
              (l.strip().split(":") for l in open(jul, encoding="utf-8") if l.strip())}
    return chave, resp, claude


def hibrido():
    """Mede o metodo hibrido (opcao 3 do responsavel, 2026-10-19): o conteudo decide quando a margem entre o galho
    mais parecido com o documento e o segundo e >= M; senao o modelo (confianca >= 0,70) ou o costume da casa."""
    import torch
    import galho_termos_soltos as GT
    from arca import decs_embeddings
    chave, resp, claude = respostas_humanas()
    GT.ARQ_MODELO = os.path.join(config.DADOS_LOCAIS, "modelo_galho_soltos_qwen3-8b.pkl")
    G = GT.Galhos()
    cams = sorted({c for l in chave.values() for c in G.vocab.caminhos(l["ui"])})
    vc = torch.nn.functional.normalize(decs_embeddings.embutir([c.replace("::", " > ") for c in cams]).float(), dim=1)
    vcam = {c: vc[i] for i, c in enumerate(cams)}
    decs_embeddings.liberar()
    casos = []
    for l in chave.values():
        s = sorted(((float(vcam[c] @ G.vet[l["handle"]]), c) for c in G.vocab.caminhos(l["ui"])), reverse=True)
        casos.append(dict(n=l["n"], margem=s[0][0] - s[1][0], A=l["metodos"]["A"], B=l["metodos"]["B"],
                          C=l["metodos"]["C"], confB=l["confianca_B"]))
    json.dump(casos, open(os.path.join(PASTA, "teste_cego_margens.json"), "w", encoding="utf-8"), indent=1)
    linhas = ["# Metodo hibrido de galho: medicao no teste cego (%s)" % datetime_hoje(), "",
              "Responsavel: %d casos julgados (aceita mais de uma opcao). Claude: %d casos (uma opcao)." % (len(resp), len(claude)),
              "Regra: margem do conteudo >= M -> conteudo; senao, modelo se confianca >= 0,70 (ou nunca), senao costume.", "",
              "| M | modelo no meio | casos pelo conteudo | acerto responsavel | acerto Claude |", "|---|---|---|---|---|"]
    for usa_b in (True, False):
        for M in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 1.0):
            def galho(c):
                if c["margem"] >= M:
                    return c["C"]
                return c["B"] if usa_b and c["confB"] >= 0.70 else c["A"]
            nc = sum(c["margem"] >= M for c in casos)
            ar = sum(galho(c) in resp[c["n"]] for c in casos if c["n"] in resp)
            ac = sum(galho(c) == claude[c["n"]] for c in casos if c["n"] in claude)
            linhas.append("| %s | %s | %d | %d/%d (%.0f%%) | %d/%d (%.0f%%) |" % (
                "so conteudo" if M == 0 else ("nunca" if M == 1 else "%.3f" % M), "sim" if usa_b else "nao", nc,
                ar, len(resp), 100 * ar / max(1, len(resp)), ac, len(claude), 100 * ac / max(1, len(claude))))
    marg = sorted(c["margem"] for c in casos)
    linhas += ["", "Margens (quartis): %.4f / %.4f / %.4f" % (marg[25], marg[50], marg[75])]
    out = os.path.join(PASTA, "teste_cego_hibrido.md")
    open(out, "w", encoding="utf-8").write("\n".join(linhas) + "\n")
    print("\n".join(linhas))
    print("relatorio:", out)


def datetime_hoje():
    from datetime import date
    return date.today().isoformat()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--resultado", action="store_true")
    ap.add_argument("--hibrido", action="store_true", help="mede o metodo hibrido contra os julgamentos humanos")
    a = ap.parse_args()
    hibrido() if a.hibrido else resultado() if a.resultado else montar(a.n)
