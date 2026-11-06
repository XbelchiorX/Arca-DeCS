# -*- coding: utf-8 -*-
"""Converte os galhos de dc.subject.decs que NAO existem no DeCS 2026 (o que a D8 deixou de fora) para o galho
correspondente do decs.xml v2.1 (pedido do responsavel, 2026-10-17).

Tabela de conversao (324 valores distintos, 1.840 ocorrencias): saidas/decs/raiz/conversao_inexistentes.json
  - mesmo_galho_so_raiz (266): o mesmo galho; o 1o nivel tem o nome da categoria (Ciencia da Informacao,
    Ciencias Humanas, Tecnologia...), por isso a D8 nao o reconheceu;
  - galho_mais_parecido (43): descritor ou nivel renomeado no DeCS 2026; galho com mais niveis em comum;
  - revisado_claude (15): ramo antigo extinto; galho mais usado pelos catalogadores (ou o unico).
Troca no proprio lugar (PATCH replace); se o valor novo ja existe no item, so retira o antigo (sem duplicar).
[DECS-ORIGINAL] antes da 1a gravacao (regra 4). Le cada item de novo antes de gravar.
Uso: python ferramentas/decs/corrigir_galhos_inexistentes_decs.py [--gravar] [--item icict/123]
"""
import argparse
import csv
import json
import os
import sys
import threading
from collections import Counter
from datetime import datetime

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402

PASTA = os.path.join(config.SAIDAS_DECS, "raiz")
CONV = json.load(open(os.path.join(PASTA, "conversao_inexistentes.json"), encoding="utf-8"))


def corrigir(cli, obj, gravar):
    md = obj.get("metadata") or {}
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    ops, linhas = [], []
    for i, val in enumerate(atuais):
        if val in CONV:
            novo = CONV[val]["novo"]
            dup = novo in atuais or novo in [l["depois"] for l in linhas]
            linhas.append({"handle": obj.get("handle"), "indice": i, "antes": val, "depois": novo,
                           "metodo": CONV[val]["metodo"], "acao": "retirar_duplicado" if dup else "trocar"})
    if not linhas:
        return "sem_alteracao", linhas
    if not gravar:
        return "a_corrigir", linhas
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(obj["handle"])
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else atuais)))
    for l in linhas:
        if l["acao"] == "trocar":
            ops.append({"op": "replace", "path": "/metadata/%s/%d/value" % (config.CAMPO_DECS, l["indice"]),
                        "value": l["depois"]})
    for l in sorted((l for l in linhas if l["acao"] == "retirar_duplicado"), key=lambda l: -l["indice"]):
        ops.append({"op": "remove", "path": "/metadata/%s/%d" % (config.CAMPO_DECS, l["indice"])})
    cli.patch("/core/items/" + obj["uuid"], ops)
    esperado = list(atuais)
    for l in linhas:
        if l["acao"] == "trocar":
            esperado[l["indice"]] = l["depois"]
    for l in sorted((l for l in linhas if l["acao"] == "retirar_duplicado"), key=lambda l: -l["indice"]):
        del esperado[l["indice"]]
    depois = [x.get("value", "") for x in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    R.registrar(obj["handle"], atuais, depois, [{"acao": "ATUALIZAR", "antes": l["antes"], "depois": l["depois"]}
                                                for l in linhas], aplicado=True)
    return ("corrigido" if depois == esperado else "DIVERGENTE"), linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravar", action="store_true")
    ap.add_argument("--item")
    a = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from arca import dspace, idiomas as I
    if a.item:
        hs = [a.item]
    else:   # os itens que a D8 listou com caminho inexistente
        rel = sorted(f for f in os.listdir(PASTA) if f.startswith("gravacao_") and f.endswith(".csv"))[-1]
        hs = sorted({r["handle"] for r in csv.DictReader(open(os.path.join(PASTA, rel), encoding="utf-8-sig"),
                                                         delimiter=";") if r["situacao"] == "caminho_inexistente"})
    print("%d itens (%s)" % (len(hs), "GRAVANDO" if a.gravar else "SIMULACAO"), flush=True)
    saida = os.path.join(PASTA, "inexistentes_%s_%s.csv" % ("gravacao" if a.gravar else "simulacao",
                                                            datetime.now().strftime("%Y%m%d_%H%M%S")))
    cont, acoes, trava = Counter(), Counter(), threading.Lock()

    def um(h):
        try:
            cli = I.get_thread_client()._c
            return corrigir(cli, cli.item_por_handle(h), a.gravar)
        except Exception as e:
            print("  ERRO %s: %s" % (h, str(e)[:200]), flush=True)
            return "erro", []

    with open(saida, "w", newline="", encoding="utf-8-sig") as f, ThreadPoolExecutor(8) as ex:
        w = csv.DictWriter(f, fieldnames=["handle", "indice", "antes", "depois", "metodo", "acao"], delimiter=";")
        w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(um, h) for h in hs]), 1):
            st, linhas = fut.result()
            with trava:
                cont[st] += 1
                for l in linhas:
                    w.writerow(l)
                    acoes[l["acao"]] += 1
            if k % 250 == 0:
                print("%d/%d %s" % (k, len(hs), dict(cont)), flush=True)
    del dspace
    print("FIM itens: %s | valores: %s" % (dict(cont), dict(acoes)))
    print("relatorio:", saida)


if __name__ == "__main__":
    main()
