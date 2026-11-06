# -*- coding: utf-8 -*-
"""Retira de dc.subject.decs o que nao tem lugar na arvore do DeCS (decisao do responsavel, 2026-10-17):
  - descritores de verificacao (check tags) sem arvore: Masculino, Feminino e variantes (Femea, Macho, Femenino);
  - qualificador solto, sem descritor ("genetica", "epidemiologia" sozinhos): nao e assunto por si so.
Nada vai para palavra-chave (decisao do responsavel). O DECS anterior vai para [DECS-ORIGINAL] antes da 1a gravacao
(regra 4), entao nada se perde (D0 restaura). Mapeia o ARCA ao vivo (como a opcao 1) e le cada item antes de gravar.
Uso: python ferramentas/decs/retirar_sem_arvore_decs.py [--gravar] [--item h]
"""
import argparse
import csv
import os
import sys
import threading
from collections import Counter
from datetime import datetime

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402
from arca.decs_interpretar import QUALIFICADOR_SOLTO, SEM_ARVORE, interpretar  # noqa: E402

PASTA = os.path.join(config.SAIDAS_DECS, "sem_arvore")
_V = {}


def vocab():
    if "v" not in _V:
        from arca import decs_vocabulario as V
        _V["v"] = V.carregar_vocabulario()
    return _V["v"]


def planejar(vals):
    linhas = []
    for i, v in enumerate(vals):
        if "::" in v:
            continue
        t = interpretar(v, vocab())["tipo"]
        if t == SEM_ARVORE:
            linhas.append((i, v, "sem_arvore"))
        elif t == QUALIFICADOR_SOLTO:
            linhas.append((i, v, "qualificador_solto"))
    return linhas


def corrigir(cli, h, gravar):
    obj = cli.item_por_handle(h)
    if not obj:
        return "erro_item", []
    md = obj.get("metadata") or {}
    vals = [m.get("value", "") for m in md.get(config.CAMPO_DECS) or []]
    linhas = planejar(vals)
    if not linhas or not gravar:
        return ("sem_alteracao" if not linhas else "a_corrigir"), linhas
    ops = []
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(obj["handle"])
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else vals)))
    esperado = list(vals)
    for i, _v, _t in sorted(linhas, key=lambda l: -l[0]):
        ops.append({"op": "remove", "path": "/metadata/%s/%d" % (config.CAMPO_DECS, i)})
        del esperado[i]
    cli.patch("/core/items/" + obj["uuid"], ops)
    depois = [x.get("value", "") for x in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    R.registrar(obj["handle"], vals, depois, [{"acao": "RETIRAR_" + t.upper(), "antes": v} for _i, v, t in linhas],
                aplicado=True)
    return ("corrigido" if depois == esperado else "DIVERGENTE"), linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravar", action="store_true")
    ap.add_argument("--item", nargs="+", help="um ou mais handles/UUIDs/URLs")
    a = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from arca import idiomas as I
    vocab()
    if a.item:
        hs = list(a.item)
    else:
        itens = I.mapear_objetos(I.criar_cliente(), 100, 8)
        hs = [it["handle"] for it in itens if it.get("handle") and planejar(
            [m.get("value", "") for m in (it.get("metadata") or {}).get(config.CAMPO_DECS) or []])]
        print("%d itens mapeados; %d a corrigir" % (len(itens), len(hs)), flush=True)
    os.makedirs(PASTA, exist_ok=True)
    saida = os.path.join(PASTA, "%s_%s.csv" % ("gravacao" if a.gravar else "simulacao",
                                              datetime.now().strftime("%Y%m%d_%H%M%S")))
    cont, tipos, valores, trava = Counter(), Counter(), Counter(), threading.Lock()

    def um(h):
        try:
            return h, corrigir(I.get_thread_client()._c, h, a.gravar)
        except Exception as e:
            print("  ERRO %s: %s" % (h, str(e)[:200]), flush=True)
            return h, ("erro", [])

    with open(saida, "w", newline="", encoding="utf-8-sig") as f, ThreadPoolExecutor(8) as ex:
        w = csv.writer(f, delimiter=";")
        w.writerow(["handle", "indice", "valor_retirado", "motivo"])
        for fut in as_completed([ex.submit(um, h) for h in hs]):
            h, (st, linhas) = fut.result()
            with trava:
                cont[st] += 1
                for l in linhas:
                    w.writerow([h] + list(l))
                    tipos[l[2]] += 1
                    valores[l[1]] += 1
    print("FIM itens: %s | valores: %s" % (dict(cont), dict(tipos)))
    print("mais frequentes:", valores.most_common(25))
    print("relatorio:", saida)


if __name__ == "__main__":
    main()
