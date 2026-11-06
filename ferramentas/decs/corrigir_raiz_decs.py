# -*- coding: utf-8 -*-
"""Poe a categoria de topo (a "raiz": "DOENCAS [C]::", "SAUDE PUBLICA [SP]::"...) nos valores de dc.subject.decs
gravados quando o decs.xml no ar nao tinha as 21 categorias (pedido do responsavel, 2026-10-17; opcao D8).

So mexe no valor que e o MESMO galho do decs.xml v2.1 sem a categoria (correspondencia unica, medida em 16/10:
10.248 valores em ~3,1 mil itens, nenhum ambiguo). Troca o valor no proprio lugar (PATCH replace por indice): a ordem,
o idioma e os demais valores ficam como estao. Caminho que nao existe mais no DeCS 2026 NAO e tocado (vai no
relatorio como "caminho_inexistente"). Antes da 1a gravacao no item, o DECS humano vai para [DECS-ORIGINAL]
(regra 4); dc.description.provenance nao e tocado (regra 2). Toda escrita pelo ClienteDSpace.patch (regra 3).

Uso: python ferramentas/decs/corrigir_raiz_decs.py --item icict/49699 [icict/...] [--gravar]
     python ferramentas/decs/corrigir_raiz_decs.py --todos [--gravar] [--refazer]
"""
import argparse
import csv
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402
from arca.decs_vocabulario import chave_sem_categoria  # noqa: E402

PASTA = os.path.join(config.SAIDAS_DECS, "raiz")
FEITOS = os.path.join(PASTA, "feitos.txt")
CAMPOS = ["handle", "uuid", "indice", "antes", "depois", "situacao"]
_VOCAB = {}


def vocab():
    if "v" not in _VOCAB:
        from arca import decs_vocabulario as V
        v = V.carregar_vocabulario()           # arvore de gravacao: decs.xml v2.1 (com categorias)
        mapa = defaultdict(set)
        for tn, val in v.servidor.items():
            mapa[chave_sem_categoria(val)].add(tn)
        _VOCAB["v"], _VOCAB["mapa"] = v, mapa
    return _VOCAB["v"], _VOCAB["mapa"]


def planejar(metadata):
    """[(indice, antes, depois, situacao)] para os valores do item."""
    v, mapa = vocab()
    saida = []
    for i, m in enumerate(metadata.get(config.CAMPO_DECS) or []):
        val = m.get("value", "")
        if "::" not in val or v.valor_canonico(val):
            continue
        tns = mapa.get(chave_sem_categoria(val), set())
        if len(tns) == 1:
            saida.append((i, val, v.servidor[next(iter(tns))], "sem_raiz"))
        elif len(tns) > 1:
            saida.append((i, val, "", "ambiguo"))
        else:
            # o indice acima nao cobre todas as categorias (ex.: K e L): o descritor ainda pode ter um unico caminho
            # valido que e o MESMO valor com a categoria de topo na frente (caso dos itens novos de 2026-10-20)
            from arca import decs_interpretar as I
            r = I.interpretar(val, v)
            cams = v.caminhos(r["ui"]) if r.get("ui") else []
            certos = [c for c in cams if c.split("::", 1)[-1] == val]
            if len(certos) == 1:
                saida.append((i, val, certos[0], "sem_raiz"))
                continue
            # no do caminho renomeado no DeCS 2026 (ex.: Firmicutes -> Bacillota, icict/79043): cada no vira o nome
            # atual do descritor (a BIREME guarda o antigo como sinonimo); so troca se o caminho traduzido for
            # EXATAMENTE um dos caminhos validos do descritor
            traduzido = []
            for seg in val.split("::"):
                rs = I.interpretar(seg, v)
                traduzido.append(v.nome(rs["ui"]) if rs.get("ui") and rs.get("tipo") in (I.NOME, I.SINONIMO) else seg)
            alvo = chave_sem_categoria("::".join(traduzido))
            certos = [c for c in cams if chave_sem_categoria(c) == alvo]
            if len(certos) == 1:
                saida.append((i, val, certos[0], "no_renomeado"))
            else:
                saida.append((i, val, "", "caminho_inexistente"))
    return saida


def corrigir(cli, item, gravar):
    """Le o item AGORA no servidor, planeja e (se gravar) troca os valores. Devolve (status, linhas)."""
    obj = cli.get("/core/items/" + item["uuid"]) if gravar else item
    md = obj.get("metadata") or {}
    plano = planejar(md)
    linhas = [{"handle": obj.get("handle"), "uuid": obj["uuid"], "indice": i, "antes": a, "depois": d,
               "situacao": s} for i, a, d, s in plano]
    trocas = [(i, d) for i, _a, d, s in plano if s in ("sem_raiz", "no_renomeado")]
    if not trocas:
        return ("sem_alteracao" if not plano else "so_outros_casos"), linhas
    if not gravar:
        return "a_corrigir", linhas
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    ops = []
    if R.ler_original(md) is None:                     # regra 4: o humano guardado uma vez so
        humano = R.humano_antes_da_primeira_aplicacao(obj["handle"])
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else atuais)))
    for i, d in trocas:
        ops.append({"op": "replace", "path": "/metadata/%s/%d/value" % (config.CAMPO_DECS, i), "value": d})
    cli.patch("/core/items/" + obj["uuid"], ops)
    depois = [x.get("value", "") for x in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    esperado = list(atuais)
    for i, d in trocas:
        esperado[i] = d
    R.registrar(obj["handle"], atuais, depois, [{"acao": "ATUALIZAR", "antes": a, "depois": d}
                                                for i, a, d, s in plano if s in ("sem_raiz", "no_renomeado")], aplicado=True)
    return ("corrigido" if depois == esperado else "DIVERGENTE"), linhas


def mapear(page_size, map_workers, data_inicio=None):
    """Itens do ARCA na hora, pelo MESMO mapeamento da opcao 1 (arca/idiomas.py: mapear_objetos)."""
    from arca import idiomas as I
    itens = I.mapear_objetos(I.criar_cliente(), page_size, map_workers, data_inicio=data_inicio)
    ac = {it["handle"]: it for it in itens if it.get("handle") and it.get("uuid")}
    print("%d itens mapeados." % len(ac), flush=True)
    return ac


def processar_todos(ac, process_workers, gravar, refazer=False, log=print):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from arca import idiomas as I
    os.makedirs(PASTA, exist_ok=True)
    feitos = set()
    if gravar and os.path.exists(FEITOS) and not refazer:
        feitos = set(open(FEITOS, encoding="utf-8").read().split())
    hs = [h for h in sorted(ac) if h not in feitos and planejar(ac[h].get("metadata") or {})]
    log("%d itens com valor de DeCS fora da arvore v2.1 (%s)" % (len(hs), "GRAVANDO" if gravar else "SIMULACAO"))
    rel = os.path.join(PASTA, "%s_%s.csv" % ("gravacao" if gravar else "simulacao",
                                             datetime.now().strftime("%Y%m%d_%H%M%S")))
    cont, vals, trava, t0 = Counter(), Counter(), threading.Lock(), time.time()

    def um(h):
        try:
            cli = I.get_thread_client()._c if gravar else None
            return h, corrigir(cli, ac[h], gravar)
        except Exception as e:
            log("  ERRO %s: %s" % (h, str(e)[:200]))
            return h, ("erro", [])

    with open(rel, "w", newline="", encoding="utf-8-sig") as f, \
            open(FEITOS, "a", encoding="utf-8") as fd, \
            ThreadPoolExecutor(max(1, process_workers if gravar else 1)) as ex:
        w = csv.DictWriter(f, fieldnames=CAMPOS, delimiter=";")
        w.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(um, h) for h in hs]), 1):
            h, (st, linhas) = fut.result()
            with trava:
                cont[st] += 1
                for l in linhas:
                    w.writerow(l)
                    vals[l["situacao"]] += 1
                if gravar and st in ("corrigido", "so_outros_casos", "sem_alteracao"):
                    fd.write(h + "\n")
                    fd.flush()
            if k % 500 == 0:
                log("%d/%d %s (%.0f min)" % (k, len(hs), dict(cont), (time.time() - t0) / 60))
    log("FIM itens: %s | valores: %s" % (dict(cont), dict(vals)))
    log("relatorio: %s" % rel)
    return cont, vals, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", nargs="+", help="um ou mais handles/UUIDs/URLs")
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--gravar", action="store_true")
    ap.add_argument("--refazer", action="store_true")
    a = ap.parse_args()
    from arca import dspace
    cli = dspace.conectar()
    if a.item:
        for ident in a.item:
            it = cli.item_por_handle(ident)
            if not it:
                print("%s: item nao encontrado" % ident)
                continue
            st, linhas = corrigir(cli, it, a.gravar)
            print("%s: %s" % (it.get("handle") or ident, st))
            for l in linhas:
                print("  [%s] %s" % (l["situacao"], l["antes"]))
                print("     -> %s" % (l["depois"] or "(nao muda)"))
    elif a.todos:
        processar_todos(mapear(100, 8), 8, a.gravar, refazer=a.refazer)


if __name__ == "__main__":
    main()
