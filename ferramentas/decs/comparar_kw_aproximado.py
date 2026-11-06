# -*- coding: utf-8 -*-
"""A/B: palavra-chave do autor -> DeCS, so literal (como hoje) x com o casamento aproximado
(arca/decs_palavra_chave.py: grafia, ligacao, plural, ordem, digitacao). Regra 8 do CLAUDE.md: medir antes.

Mesmo protocolo e MESMOS itens do treinar_avaliar.py (teste 1500 / treino 12000, semente 20261002) e a MESMA rede
salva na ultima avaliacao (treinada sem os itens de teste): a unica diferenca entre os dois lados e o sinal da
palavra-chave. Nao salva modelo nenhum; so mede.

Saida: saidas/avaliacao/comparacao_kw_aproximado.md / .json
Uso:   python ferramentas/decs/comparar_kw_aproximado.py [--teste 1500] [--treino 12000]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

os.environ["DECS_KW_APROXIMADO"] = "sim"          # CARACTERISTICAS com a coluna nova; o lado A a deixa zerada
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,  # noqa: E402
                           TimeElapsedColumn, TimeRemainingColumn)
from arca import config, decs_acervo, decs_indexador as IX, decs_rede  # noqa: E402
from arca import decs_vocabulario as V  # noqa: E402
from treinar_avaliar import duplicatas, metricas_ranking  # noqa: E402

COLS_NOVAS = {IX.CARACTERISTICAS.index(c) for c in ("lex_kw_aprox", "lex_kw_aprox_fraco")}


def barra(titulo):
    return Progress(TextColumn("[bold]%s" % titulo), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                    TimeRemainingColumn(), TextColumn("{task.percentage:.1f}%"))


def coletar(ix, handles, titulo):
    linhas, grupos = [], []
    with barra(titulo) as pr:
        t = pr.add_task("", total=len(handles))
        for h in handles:
            vet = ix.V[ix.pos[h]]
            for u, f in ix.caracteristicas(ix.acervo[h], vet, excluir=duplicatas(ix, h, vet), proprio=h).items():
                linhas.append(f)
                grupos.append((h, u))
            pr.advance(t)
    X = np.array(linhas, dtype=np.float32)
    y = np.array([1 if u in set(ix.gabarito[h]) else 0 for h, u in grupos], dtype=np.int8)
    return X, y, grupos


def medir(ix, teste, Xtr, ytr, Xte, yte, gte, colunas, afetados):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=0).fit(Xtr[:, colunas], ytr)
    p = m.predict_proba(Xte[:, colunas])[:, 1]
    rk = defaultdict(list)
    for (h, u), s in zip(gte, p):
        rk[h].append((u, s))
    for h in teste:
        rk[h].sort(key=lambda z: -z[1])
    res = metricas_ranking(rk, ix.gabarito, [x / 100 for x in range(20, 100)])
    out = {k: v for k, v in res.items() if k.startswith(("P@", "R@"))}
    for alvo, nome in ((config.DECS_PRECISAO_AUTOMATICA, "automatico"), (config.DECS_PRECISAO_REVISAO, "revisao")):
        ok = [(float(k[2:]), v) for k, v in res.items() if k.startswith("t=") and v["precisao"] >= alvo]
        t, v = min(ok) if ok else (0.99, {"precisao": 0, "recall": 0, "f1": 0, "por_item": 0})
        out[nome] = dict(v, limiar=t)
    # recall maximo dos candidatos (quantos descritores humanos chegam a ser candidatos)
    out["recall_maximo"] = float(yte.sum() / sum(len(ix.gabarito[h]) for h in teste))
    # so nos itens de teste em que o casamento aproximado age (alguma palavra-chave so casa por ele)
    sub = {h: rk[h] for h in teste if h in afetados}
    rs = metricas_ranking(sub, ix.gabarito, [x / 100 for x in range(20, 100)])
    out["afetados"] = {k: v for k, v in rs.items() if k.startswith(("P@", "R@"))}
    out["afetados"]["n"] = len(sub)
    for alvo, nome in ((config.DECS_PRECISAO_AUTOMATICA, "automatico"), (config.DECS_PRECISAO_REVISAO, "revisao")):
        t = out[nome]["limiar"]                   # o limiar do acervo inteiro, aplicado ao recorte
        out["afetados"][nome] = rs["t=%.2f" % t]
    return out


def diagnostico(ix, handles):
    """Por nivel: quantas palavras-chave casam e quantas apontam um descritor que o catalogador pos."""
    cont, certo, exemplos, afetados = Counter(), Counter(), defaultdict(list), set()
    aprox = IX._indice_aproximado(ix.vocab)
    for h in handles:
        g = set(ix.gabarito.get(h, ()))
        for kw in IX._kw_textos(ix.acervo[h]):
            uis, nivel = ix.vocab.uis_do_termo(kw)
            if nivel in ("nome", "sinonimo"):
                cont[nivel] += 1
                certo[nivel] += bool(g & set(uis))
                continue
            if nivel == "variante":
                cont["variante (hoje)"] += 1
                certo["variante (hoje)"] += bool(g & set(uis))
            u2, n2 = aprox.uis(kw)
            n2 = n2 or "nao casa"
            cont[n2] += 1
            ok = bool(g & set(u2))
            if u2:
                afetados.add(h)
            certo[n2] += ok
            if u2 and len(exemplos[n2]) < 40:
                exemplos[n2].append((kw, ix.vocab.nome(u2[0]), "certo" if ok else "catalogador nao pos"))
    return {n: {"palavras_chave": cont[n], "catalogador_pos": certo[n],
                "taxa": certo[n] / max(1, cont[n])} for n in cont}, exemplos, afetados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", type=int, default=1500)
    ap.add_argument("--treino", type=int, default=12000)
    args = ap.parse_args()
    t0 = time.time()
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    acervo = decs_acervo.carregar_acervo()
    ix = IX.montar_indexador(vocab, acervo, com_modelo=False)
    eleg = sorted(h for h in ix.handles if ix.gabarito.get(h) and decs_acervo.tem_texto_suficiente(acervo[h]))
    random.seed(20261002)
    random.shuffle(eleg)
    teste, treino = eleg[:args.teste], eleg[args.teste:args.teste + args.treino]
    ant = json.load(open(os.path.join(config.SAIDAS_AVALIACAO, "avaliacao.json"), encoding="utf-8"))
    mesmo = ant.get("teste_handles") == teste
    print("itens de teste iguais aos da avaliacao de %s: %s" % (ant.get("data"), mesmo), flush=True)
    if not mesmo:
        sys.exit("Os itens de teste mudaram (acervo diferente?): a rede salva poderia ter visto o teste. Parei.")
    ix.recalcular_prior(excluir=teste)
    ix.rede = decs_rede.carregar()
    print("indexador + rede salva prontos (%.0fs)" % (time.time() - t0), flush=True)

    diag, exemplos, afetados = diagnostico(ix, treino + teste)
    print("  itens de teste em que o casamento aproximado age: %d" % len(afetados & set(teste)), flush=True)
    for n, d in sorted(diag.items(), key=lambda z: -z[1]["palavras_chave"]):
        print("  %-16s %6d palavras-chave; catalogador pos o descritor em %.1f%%" % (
            n, d["palavras_chave"], 100 * d["taxa"]), flush=True)

    lados = {}
    cols_a = [i for i in range(len(IX.CARACTERISTICAS)) if i not in COLS_NOVAS]
    for nome, ligado in (("A. so literal (hoje)", False), ("B. + casamento aproximado", True)):
        config.DECS_KW_APROXIMADO = ligado
        Xtr, ytr, _ = coletar(ix, treino, "%s: treino" % nome[:2])
        Xte, yte, gte = coletar(ix, teste, "%s: teste" % nome[:2])
        cols = list(range(len(IX.CARACTERISTICAS))) if ligado else cols_a
        lados[nome] = medir(ix, teste, Xtr, ytr, Xte, yte, gte, cols, afetados)
        r = lados[nome]
        print("%s | P@1 %.3f P@5 %.3f R@10 %.3f | auto: limiar %.2f prec %.3f recall %.3f | revisao: recall %.3f"
              % (nome, r["P@1"], r["P@5"], r["R@10"], r["automatico"]["limiar"], r["automatico"]["precisao"],
                 r["automatico"]["recall"], r["revisao"]["recall"]), flush=True)
        a = r["afetados"]
        print("   so os %d afetados | P@1 %.3f P@5 %.3f R@10 %.3f | auto: prec %.3f recall %.3f | revisao: prec %.3f "
              "recall %.3f" % (a["n"], a["P@1"], a["P@5"], a["R@10"], a["automatico"]["precisao"],
                               a["automatico"]["recall"], a["revisao"]["precisao"], a["revisao"]["recall"]), flush=True)

    res = {"data": datetime.now().isoformat(timespec="seconds"), "n_teste": len(teste), "n_treino": len(treino),
           "lados": lados, "diagnostico_palavras_chave": diag, "exemplos": exemplos}
    base = os.path.join(config.SAIDAS_AVALIACAO, "comparacao_kw_aproximado")
    json.dump(res, open(base + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    L = ["# Palavra-chave do autor -> DeCS: so literal x casamento aproximado", "",
         "Gerado em %s. Teste %d itens (os mesmos da avaliacao de %s), treino %d. Mesma rede, mesmo kNN; muda so o "
         "sinal da palavra-chave." % (res["data"], len(teste), ant.get("data"), len(treino)), "",
         "| Lado | P@1 | P@3 | P@5 | R@5 | R@10 | Recall maximo | Limiar auto | Precisao | Recall auto | "
         "Recall revisao (70%) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nome, r in lados.items():
        L.append("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.2f | %.1f%% | %.1f%% | %.1f%% |" % (
            nome, 100 * r["P@1"], 100 * r["P@3"], 100 * r["P@5"], 100 * r["R@5"], 100 * r["R@10"],
            100 * r["recall_maximo"], r["automatico"]["limiar"], 100 * r["automatico"]["precisao"],
            100 * r["automatico"]["recall"], 100 * r["revisao"]["recall"]))
    L += ["", "## So os itens de teste em que o casamento aproximado age", "",
          "| Lado | Itens | P@1 | P@5 | R@10 | Precisao auto | Recall auto | Precisao revisao | Recall revisao |",
          "|---|---|---|---|---|---|---|---|---|"]
    for nome, r in lados.items():
        a = r["afetados"]
        L.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
            nome, a["n"], 100 * a["P@1"], 100 * a["P@5"], 100 * a["R@10"], 100 * a["automatico"]["precisao"],
            100 * a["automatico"]["recall"], 100 * a["revisao"]["precisao"], 100 * a["revisao"]["recall"]))
    L += ["", "## Palavras-chave do treino+teste por nivel de casamento", "",
          "| Nivel | Palavras-chave | Catalogador pos o descritor |", "|---|---|---|"]
    for n, d in sorted(diag.items(), key=lambda z: -z[1]["palavras_chave"]):
        L.append("| %s | %d | %.1f%% |" % (n, d["palavras_chave"], 100 * d["taxa"]))
    for n, ex in exemplos.items():
        L += ["", "### Exemplos: %s" % n, ""] + ["- %s -> %s (%s)" % e for e in ex[:25]]
    open(base + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("relatorio: %s.md (%.0fs)" % (base, time.time() - t0))


if __name__ == "__main__":
    main()
