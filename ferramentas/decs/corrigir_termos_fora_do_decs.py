# -*- coding: utf-8 -*-
"""Corrige os valores de dc.subject.decs que nao sao do DeCS 2026 (pedido e aprovacao do responsavel, 2026-10-17).

Decisoes do Claude, termo a termo (3.330 termos distintos), em saidas/decs/nao_decs/plano_nao_decs.json
(planilha: revisao_termos_fora_do_decs.xlsx). Criterios: converter so quando e o MESMO conceito; generico ou fora do
DeCS vira palavra-chave; lixo sai; duvida ("revisar") NAO e tocada.
  - converter: troca o valor no proprio lugar (galho unico: ja com o caminho; varios galhos: o nome do descritor,
    termo solto, para a etapa dos galhos). Se o descritor ja esta no item, so retira (nao duplica);
  - palavra-chave: sai do DeCS e entra em dc.subject.other, salvo se o termo (sem acento/caixa) ja esta em
    dc.subject.other/.en/.es/.fr;
  - retirar: sai do DeCS.
[DECS-ORIGINAL] antes da 1a gravacao (regra 4); le cada item AO VIVO antes de gravar; confere depois.
Uso: python ferramentas/decs/corrigir_termos_fora_do_decs.py [--gravar] [--item icict/123]
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
from arca.decs_vocabulario import norm  # noqa: E402

PASTA = os.path.join(config.SAIDAS_DECS, "nao_decs")
PLANO = {r["termo"]: r for r in json.load(open(os.path.join(PASTA, "plano_nao_decs.json"), encoding="utf-8"))
         if r["acao"] != "revisar"}
CAMPO_KW = config.CAMPO_PALAVRA_CHAVE
CAMPOS_KW_TODOS = ["dc.subject.other", "dc.subject.en", "dc.subject.es", "dc.subject.fr"]
_V = {}


def vocab():
    if "v" not in _V:
        from arca import decs_vocabulario as V
        _V["v"] = V.carregar_vocabulario()
    return _V["v"]


def ui_do_valor(val):
    from arca.decs_interpretar import interpretar
    return interpretar(val, vocab()).get("ui")


def decisao(valor):
    """Decisao para o valor: a do plano (termo a termo, 2026-10-17) ou, para termo novo que a BIREME nao reconhece
    de jeito nenhum (nem nome, sinonimo ou caminho), a regra geral aprovada: fora do DeCS vira palavra-chave."""
    if valor in PLANO:
        return PLANO[valor]
    if "::" in valor or not valor.strip():
        return None
    from arca.decs_interpretar import interpretar
    if interpretar(valor, vocab()).get("tipo") == "nao_encontrado":
        return {"acao": "palavra-chave"}
    return None


def planejar(md):
    vals = md.get(config.CAMPO_DECS) or []
    atuais = [m.get("value", "") for m in vals]
    planos = {a: decisao(a) for a in atuais}
    if not any(planos.values()):
        return []
    uis_outros = {ui_do_valor(a) for a in atuais if not planos[a]} - {None}
    kws = {norm(m.get("value", "")) for c in CAMPOS_KW_TODOS for m in md.get(c) or []}
    linhas, novos_uis, novas_kw = [], set(), set()
    for i, (m, a) in enumerate(zip(vals, atuais)):
        r = planos[a]
        if not r:
            continue
        if r["acao"] == "converter":
            if r["ui"] in uis_outros or r["ui"] in novos_uis:
                linhas.append((i, a, "", "retirar_duplicado", r["descritor"]))
            else:
                novos_uis.add(r["ui"])
                linhas.append((i, a, r["grava"], "converter", r["descritor"]))
        elif r["acao"] == "palavra-chave":
            k = norm(a)
            if k in kws or k in novas_kw:
                linhas.append((i, a, "", "palavra_chave_ja_existia", m.get("language") or "pt-BR"))
            else:
                novas_kw.add(k)
                linhas.append((i, a, a, "palavra_chave_nova", m.get("language") or "pt-BR"))
        else:
            linhas.append((i, a, "", "retirar", ""))
    return linhas


def corrigir(cli, h, gravar):
    obj = cli.item_por_handle(h)
    if not obj:
        return "erro_item", []
    md = obj.get("metadata") or {}
    linhas = planejar(md)
    if not linhas or not gravar:
        return ("sem_alteracao" if not linhas else "a_corrigir"), linhas
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    ops = []
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(obj["handle"])
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else atuais)))
    esperado = list(atuais)
    for i, _a, novo, acao, _x in linhas:
        if acao == "converter":
            ops.append({"op": "replace", "path": "/metadata/%s/%d/value" % (config.CAMPO_DECS, i), "value": novo})
            esperado[i] = novo
    for i, _a, _n, acao, _x in sorted(linhas, key=lambda l: -l[0]):
        if acao != "converter":
            ops.append({"op": "remove", "path": "/metadata/%s/%d" % (config.CAMPO_DECS, i)})
            del esperado[i]
    for _i, _a, novo, acao, lang in linhas:
        if acao == "palavra_chave_nova":
            ops.append({"op": "add", "path": "/metadata/%s/-" % CAMPO_KW, "value": {"value": novo, "language": lang}})
    cli.patch("/core/items/" + obj["uuid"], ops)
    depois = [x.get("value", "") for x in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    R.registrar(obj["handle"], atuais, depois, [{"acao": l[3].upper(), "antes": l[1], "depois": l[2]} for l in linhas],
                aplicado=True)
    return ("corrigido" if depois == esperado else "DIVERGENTE"), linhas


def mapear():
    """Itens do ARCA na hora, pelo MESMO mapeamento da opcao 1; so os que tem algum termo do plano."""
    from arca import idiomas as I
    itens = I.mapear_objetos(I.criar_cliente(), 100, 8)
    hs = [it["handle"] for it in itens if it.get("handle") and any(
        m.get("value", "") in PLANO for m in (it.get("metadata") or {}).get(config.CAMPO_DECS) or [])]
    print("%d itens mapeados; %d com termo fora do DeCS a corrigir" % (len(itens), len(hs)), flush=True)
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravar", action="store_true")
    ap.add_argument("--item", nargs="+")
    a = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from arca import idiomas as I
    vocab()
    hs = list(a.item) if a.item else mapear()
    saida = os.path.join(PASTA, "%s_%s.csv" % ("gravacao" if a.gravar else "simulacao",
                                              datetime.now().strftime("%Y%m%d_%H%M%S")))
    cont, acoes, trava = Counter(), Counter(), threading.Lock()

    def um(h):
        try:
            return h, corrigir(I.get_thread_client()._c, h, a.gravar)
        except Exception as e:
            print("  ERRO %s: %s" % (h, str(e)[:200]), flush=True)
            return h, ("erro", [])

    with open(saida, "w", newline="", encoding="utf-8-sig") as f, ThreadPoolExecutor(8) as ex:
        w = csv.writer(f, delimiter=";")
        w.writerow(["handle", "indice", "antes", "depois", "acao", "descritor_ou_idioma"])
        for k, fut in enumerate(as_completed([ex.submit(um, h) for h in hs]), 1):
            h, (st, linhas) = fut.result()
            with trava:
                cont[st] += 1
                for l in linhas:
                    w.writerow([h] + list(l))
                    acoes[l[3]] += 1
            if k % 500 == 0:
                print("%d/%d %s" % (k, len(hs), dict(cont)), flush=True)
    print("FIM itens: %s | valores: %s" % (dict(cont), dict(acoes)))
    print("relatorio:", saida)


if __name__ == "__main__":
    main()
