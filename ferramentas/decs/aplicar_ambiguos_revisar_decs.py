# -*- coding: utf-8 -*-
"""Aplica as decisoes do Claude, caso a caso (lendo o documento), para os termos AMBIGUOS (servem a mais de um
descritor) e os 22 termos que tinham ficado para revisao (autorizado pelo responsavel, 2026-10-17).
Decisoes: saidas/decs/nao_decs/decisoes_ambiguos_revisar.txt (indice de casos_ambiguos_revisar.json).
  D|A;B  -> o termo vira o descritor A (no proprio lugar) e B entra no fim; galho unico ja com o caminho, varios
            galhos com o nome (termo solto, para a etapa dos galhos); descritor que ja esta no item nao duplica;
  K      -> sai do DeCS e vai para dc.subject.other (sem duplicar o que ja esta em other/en/es/fr);
  X      -> sai do DeCS (pedaco de valor cortado na virgula).
[DECS-ORIGINAL] antes da 1a gravacao; le cada item ao vivo; confere depois.
Uso: python ferramentas/decs/aplicar_ambiguos_revisar_decs.py [--gravar]
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402
from arca.decs_vocabulario import norm  # noqa: E402

PASTA = os.path.join(config.SAIDAS_DECS, "nao_decs")
CAMPOS_KW = ["dc.subject.other", "dc.subject.en", "dc.subject.es", "dc.subject.fr"]


def carregar():
    from arca import decs_vocabulario as V
    v = V.carregar_vocabulario()
    casos = json.load(open(os.path.join(PASTA, "casos_ambiguos_revisar.json"), encoding="utf-8"))
    por_item = defaultdict(dict)
    for l in open(os.path.join(PASTA, "decisoes_ambiguos_revisar.txt"), encoding="utf-8"):
        p = l.strip().split("|")
        if not p[0].isdigit():
            continue
        c = casos[int(p[0]) - 1]
        alvos = []
        if p[1] == "D":
            for t in p[2].split(";"):
                u = t if t in v.base else [x for x in v.uis_do_termo(t)[0] if v.caminhos(x)][0]
                cams = v.caminhos(u)
                alvos.append((u, cams[0] if len(cams) == 1 else v.nome(u)))
        por_item[c["handle"]][c["valor"]] = (p[1], alvos)
    return v, por_item


def corrigir(cli, v, h, dec, gravar):
    from arca.decs_interpretar import interpretar
    obj = cli.item_por_handle(h)
    md = obj.get("metadata") or {}
    vals = [m.get("value", "") for m in md.get(config.CAMPO_DECS) or []]
    langs = [m.get("language") or "pt-BR" for m in md.get(config.CAMPO_DECS) or []]
    uis = {interpretar(x, v).get("ui") for x in vals if x not in dec} - {None}
    kws = {norm(m.get("value", "")) for c in CAMPOS_KW for m in md.get(c) or []}
    trocas, remover, adicionar, novas_kw, linhas = {}, [], [], [], []
    for i, x in enumerate(vals):
        if x not in dec:
            continue
        acao, alvos = dec[x]
        if acao == "D":
            primeiro = True
            for u, valor in alvos:
                if u in uis:
                    linhas.append((x, valor, "ja_no_item"))
                    continue
                uis.add(u)
                if primeiro:
                    trocas[i] = valor
                    primeiro = False
                    linhas.append((x, valor, "converter"))
                else:
                    adicionar.append(valor)
                    linhas.append((x, valor, "acrescentar"))
            if primeiro:
                remover.append(i)
        elif acao == "K":
            remover.append(i)
            if norm(x) not in kws:
                kws.add(norm(x))
                novas_kw.append((x, langs[i]))
                linhas.append((x, x, "palavra_chave_nova"))
            else:
                linhas.append((x, "", "palavra_chave_ja_existia"))
        else:
            remover.append(i)
            linhas.append((x, "", "retirar"))
    if not linhas or not gravar:
        return ("sem_alteracao" if not linhas else "a_corrigir"), linhas
    ops = []
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(h)
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else vals)))
    esperado = list(vals)
    for i, valor in trocas.items():
        ops.append({"op": "replace", "path": "/metadata/%s/%d/value" % (config.CAMPO_DECS, i), "value": valor})
        esperado[i] = valor
    for i in sorted(set(remover), reverse=True):
        ops.append({"op": "remove", "path": "/metadata/%s/%d" % (config.CAMPO_DECS, i)})
        del esperado[i]
    for valor in adicionar:
        ops.append({"op": "add", "path": "/metadata/%s/-" % config.CAMPO_DECS, "value": {"value": valor, "language": "pt-BR"}})
        esperado.append(valor)
    for x, lang in novas_kw:
        ops.append({"op": "add", "path": "/metadata/%s/-" % config.CAMPO_PALAVRA_CHAVE, "value": {"value": x, "language": lang}})
    cli.patch("/core/items/" + obj["uuid"], ops)
    depois = [m.get("value", "") for m in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    R.registrar(h, vals, depois, [{"acao": a.upper(), "antes": x, "depois": d} for x, d, a in linhas], aplicado=True)
    return ("corrigido" if depois == esperado else "DIVERGENTE"), linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravar", action="store_true")
    a = ap.parse_args()
    from arca import dspace
    v, por_item = carregar()
    cli = dspace.conectar()
    saida = os.path.join(PASTA, "ambiguos_revisar_%s_%s.csv" % ("gravacao" if a.gravar else "simulacao",
                                                               datetime.now().strftime("%Y%m%d_%H%M%S")))
    cont, acoes = Counter(), Counter()
    with open(saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["handle", "antes", "depois", "acao"])
        for h, dec in sorted(por_item.items()):
            try:
                st, linhas = corrigir(cli, v, h, dec, a.gravar)
            except Exception as e:
                st, linhas = "erro", []
                print("  ERRO %s: %s" % (h, str(e)[:200]))
            cont[st] += 1
            for l in linhas:
                w.writerow([h] + list(l))
                acoes[l[2]] += 1
    print("FIM itens: %s | valores: %s" % (dict(cont), dict(acoes)))
    print("relatorio:", saida)


if __name__ == "__main__":
    main()
