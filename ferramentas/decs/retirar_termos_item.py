# -*- coding: utf-8 -*-
"""Retira de um ou mais itens os descritores DeCS indicados pelo nome (o ultimo no do caminho), em qualquer galho.
Le cada item AO VIVO, retira so os valores daquele descritor, confere depois e registra no SQLite. Como fica
registrado, o D1/D2, a rotina e a gravacao em massa nao recolocam o termo (R.descritores_retirados).
[DECS-ORIGINAL] antes da 1a gravacao (regra 4); toda escrita pelo ClienteDSpace.patch (regra 3).
Uso: python ferramentas/decs/retirar_termos_item.py --item icict/79038 --termo "Eugenia" --termo "Política de Saúde" [--gravar]
"""
import argparse
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402
from arca.decs_vocabulario import norm  # noqa: E402


def retirar(cli, ident, termos, gravar):
    obj = cli.item_por_handle(ident)
    if not obj:
        return "item nao encontrado", []
    md = obj.get("metadata") or {}
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    alvo = {norm(t) for t in termos}
    idx = [i for i, v in enumerate(atuais) if norm(v.split("::")[-1]) in alvo]
    if not idx:
        return "nada a retirar", []
    sai = [atuais[i] for i in idx]
    if not gravar:
        return "retiraria", sai
    ops = []
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(obj["handle"])
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else atuais)))
    ops += [{"op": "remove", "path": "/metadata/%s/%d" % (config.CAMPO_DECS, i)} for i in sorted(idx, reverse=True)]
    esperado = [v for i, v in enumerate(atuais) if i not in idx]
    if not esperado:
        ops.extend(R.ops_remover_linhas(md, (R.PREFIXO_AUTO,)))
    cli.patch("/core/items/" + obj["uuid"], ops)
    depois = [x.get("value", "") for x in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    R.registrar(obj["handle"], atuais, depois, [{"acao": "RETIRAR", "antes": v, "depois": "",
                                                 "motivo": "retirado a pedido (retirar_termos_item.py)"} for v in sai],
                aplicado=True)
    return ("retirado" if depois == esperado else "DIVERGENTE"), sai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", nargs="+", required=True, help="handles, UUIDs ou URLs")
    ap.add_argument("--termo", action="append", required=True, help="nome do descritor (pode repetir)")
    ap.add_argument("--gravar", action="store_true")
    a = ap.parse_args()
    from arca import dspace
    cli = dspace.conectar()
    for ident in a.item:
        st, sai = retirar(cli, ident, a.termo, a.gravar)
        print("%s: %s" % (ident, st))
        for v in sai:
            print("   - %s" % v)


if __name__ == "__main__":
    main()
