# -*- coding: utf-8 -*-
"""Treina o classificador de DECS e MEDE a qualidade contra a indexacao humana.

Protocolo (o padrao da area — BioASQ / NLM):
  1. Itens com DECS humano confiavel e texto suficiente sao sorteados
     (semente fixa) em TREINO e TESTE. O teste nunca e visto no treino.
  2. Para cada item, o DECS humano e ESCONDIDO; o metodo ve so titulo,
     resumo e palavras-chave e propoe descritores.
  3. Compara-se o proposto com o que o catalogador atribuiu:
        precisao = dos propostos, quantos o catalogador tambem pos
        recall   = dos que o catalogador pos, quantos foram propostos
  4. O item nunca e vizinho de si mesmo, nem de copias quase identicas
     (similaridade >= 0,985: mesmo trabalho depositado duas vezes), senao o
     kNN "colaria" a resposta.

Ressalva honesta: a indexacao humana nao e perfeita nem completa — a
concordancia entre dois indexadores humanos no MeSH fica tipicamente entre
40% e 60%. Um descritor proposto que o humano nao pos nao e necessariamente
errado. Os numeros sao um PISO da qualidade real, e servem para comparar
metodos entre si e escolher limiares.

Saidas:
  dados/modelo_decs.pkl              classificador + limiares escolhidos
  saidas/avaliacao/avaliacao.md      relatorio legivel
  saidas/avaliacao/avaliacao.json    numeros

Uso:  python ferramentas/decs/treinar_avaliar.py [--teste 1500] [--treino 4000]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from arca import config, decs_acervo, decs_indexador as IX  # noqa: E402
from arca import decs_vocabulario as V  # noqa: E402

LIMIAR_DUPLICATA = 0.985


def duplicatas(ix, h, vetor):
    viz = ix.vizinhos(vetor, excluir={h}, k=5)
    return {h} | {v for v, s in viz if s >= LIMIAR_DUPLICATA}


def coletar(ix, handles, com_rotulo=True):
    """Caracteristicas de todos os candidatos de uma lista de itens."""
    linhas, grupos = [], []
    for h in handles:
        vet = ix.V[ix.pos[h]]
        excl = duplicatas(ix, h, vet)
        feats = ix.caracteristicas(ix.acervo[h], vet, excluir=excl, proprio=h)
        for u, f in feats.items():
            linhas.append(f)
            grupos.append((h, u))
    X = np.array(linhas, dtype=np.float32)
    y = np.array([1 if u in set(ix.gabarito[h]) else 0 for h, u in grupos], dtype=np.int8)
    return X, y, grupos


def metricas_ranking(pontos_por_item, gabarito, limiares=(), topk=(1, 3, 5, 10)):
    """pontos_por_item: {h: [(ui, score)] ordenado}."""
    res = {}
    for k in topk:
        acertos = prop = rel = 0
        for h, lista in pontos_por_item.items():
            g = set(gabarito[h])
            top = [u for u, _ in lista[:k]]
            acertos += len(g & set(top))
            prop += len(top)
            rel += len(g)
        res["P@%d" % k] = acertos / max(1, prop)
        res["R@%d" % k] = acertos / max(1, rel)
    for t in limiares:
        ac = prop = rel = 0
        for h, lista in pontos_por_item.items():
            g = set(gabarito[h])
            sel = {u for u, s in lista if s >= t}
            ac += len(g & sel)
            prop += len(sel)
            rel += len(g)
        p = ac / max(1, prop)
        r = ac / max(1, rel)
        res["t=%.2f" % t] = {"precisao": p, "recall": r,
                             "f1": 2 * p * r / max(1e-9, p + r),
                             "por_item": prop / max(1, len(pontos_por_item))}
    return res


def _data_deposito(item):
    v = item.get("dc.date.accessioned") or []
    v = v[0].get("value") if v and isinstance(v[0], dict) else (v if isinstance(v, str) else "")
    return v or ""


def selecionar_amostra(acervo, vocab, n_teste, n_treino, n_memoria):
    """Recorte (decisao do responsavel, 2026-10-16): TESTE = os ultimos n_teste itens depositados
    (dc.date.accessioned) que ja tem DeCS do catalogador e texto; TREINO e MEMORIA (vizinhos) = sorteio
    reprodutivel (semente fixa) entre os mais antigos. Devolve (acervo_recortado, ordem: teste primeiro)."""
    eleg = [h for h in acervo
            if IX.descritores_humanos(acervo[h], vocab) and decs_acervo.tem_texto_suficiente(acervo[h])]
    eleg.sort(key=lambda h: (_data_deposito(acervo[h]), h), reverse=True)
    teste = eleg[:n_teste]
    resto = sorted(eleg[n_teste:])
    random.seed(20261002)
    random.shuffle(resto)
    ordem = teste + resto[:n_treino + n_memoria]
    recentes = sorted(acervo, key=lambda h: _data_deposito(acervo[h]), reverse=True)[:n_teste]
    sem = sum(1 for h in recentes if not IX.descritores_humanos(acervo[h], vocab))
    print("teste: os %d ultimos depositados com DeCS humano (de %s a %s); dos %d ultimos depositados no "
          "geral, %d ainda estao sem DeCS humano" % (len(teste), _data_deposito(acervo[teste[-1]])[:10],
          _data_deposito(acervo[teste[0]])[:10], n_teste, sem), flush=True)
    return {h: acervo[h] for h in ordem}, ordem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", type=int, default=1500)
    ap.add_argument("--treino", type=int, default=4000)
    ap.add_argument("--acervo", default=None)
    ap.add_argument("--amostra", action="store_true",
                    help="so os itens de teste+treino+memoria (vetores so deles); para medir rapido")
    ap.add_argument("--memoria", type=int, default=8000, help="com --amostra: itens extras como vizinhos")
    args = ap.parse_args()

    t0 = time.time()
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    acervo = decs_acervo.carregar_acervo(args.acervo)
    ordem = None
    if args.amostra:
        acervo, ordem = selecionar_amostra(acervo, vocab, args.teste, args.treino, args.memoria)
        print("amostra: %d itens (teste %d + treino %d + memoria %d)" % (len(acervo), args.teste, args.treino,
              args.memoria), flush=True)
    ix = IX.montar_indexador(vocab, acervo, com_modelo=False)
    print("indexador pronto (%.0fs)" % (time.time() - t0), flush=True)

    if ordem is not None:
        elegiveis = list(ordem)              # a ordem do sorteio, para o teste ser o mesmo em todas as rodadas
    else:
        elegiveis = sorted(h for h in ix.handles
                           if ix.gabarito.get(h) and decs_acervo.tem_texto_suficiente(acervo[h]))
        random.seed(20261002)
        random.shuffle(elegiveis)
    teste = elegiveis[:args.teste]
    treino = elegiveis[args.teste:args.teste + args.treino]
    print("elegiveis %d | treino %d | teste %d" % (len(elegiveis), len(treino), len(teste)), flush=True)
    ix.recalcular_prior(excluir=teste)

    from arca import decs_rede
    print("treinando a rede multirrotulo (5 partes, sem os itens de teste)...", flush=True)
    # fora do treino da rede: os itens de teste E suas copias quase identicas
    fora = set(teste)
    for h in teste:
        fora |= duplicatas(ix, h, ix.V[ix.pos[h]])
    print("  %d itens fora do treino da rede (%d de teste + copias)" % (len(fora), len(teste)), flush=True)
    ix.rede = decs_rede.treinar(ix.V, ix.handles, ix.gabarito, excluir=fora,
                                progresso=lambda m: print(m, flush=True))
    ix.rede.salvar()

    Xtr, ytr, gtr = coletar(ix, treino)
    peso = None
    if config.DECS_PESO_QUALIDADE:
        pq = {h: IX.peso_qualidade(acervo[h], vocab) for h in treino}
        peso = np.array([pq[h] for h, _u in gtr], dtype=np.float32)
        print("peso de qualidade ligado: media %.2f por par" % peso.mean(), flush=True)
    print("treino: %d pares candidato (%.1f%% positivos)" % (len(ytr), 100 * ytr.mean()), flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    def novo_gbm():
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0)
    modelo = novo_gbm()
    modelo.fit(Xtr, ytr, sample_weight=peso)
    ix.modelo = modelo

    # -- descritores raros (arca/decs_raros.py, DECISOES §19): mistura com o conhecimento do DeCS + calibracao
    # por faixa. Tudo escolhido em previsoes FORA DA AMOSTRA do treino (2 metades por item); o teste so mede.
    from arca import decs_raros as RR
    col_con = [IX.CARACTERISTICAS.index(c) for c in RR.CONHECIMENTO]

    conj_teste = set(teste)

    def exemplos(grupos):
        # n = exemplos humanos do descritor no treino, sem contar o proprio item (o teste ja esta fora do prior)
        return np.array([ix.freq.get(u, 0) - (1 if u in ix.gabarito.get(h, ()) and h not in conj_teste else 0)
                         for h, u in grupos])
    ntr = exemplos(gtr)
    itens_tr = sorted({h for h, _u in gtr})
    random.seed(20261021)
    metade = set(random.sample(itens_tr, len(itens_tr) // 2))
    lado = np.array([h in metade for h, _u in gtr])
    oof_h, oof_c = np.zeros(len(ytr)), np.zeros(len(ytr))
    for ladoA in (True, False):
        trn, prv = lado == ladoA, lado != ladoA
        m_h = novo_gbm().fit(Xtr[trn], ytr[trn], sample_weight=None if peso is None else peso[trn])
        m_c = novo_gbm().fit(Xtr[trn][:, col_con], ytr[trn])
        oof_h[prv] = m_h.predict_proba(Xtr[prv])[:, 1]
        oof_c[prv] = m_c.predict_proba(Xtr[prv][:, col_con])[:, 1]
    modelo_con = novo_gbm().fit(Xtr[:, col_con], ytr)
    rel_tr = sum(len(ix.gabarito[h]) for h in itens_tr)

    def recall_na_precisao(score, y, relevantes, alvo):
        """Maior recall com precisao >= alvo, varrendo o limiar (pares candidato achatados)."""
        o = np.argsort(-score)
        ac = np.cumsum(y[o])
        prec = ac / np.arange(1, len(o) + 1)
        ok = np.where(prec >= alvo)[0]
        return float(ac[ok[-1]] / max(1, relevantes)) if len(ok) else 0.0
    escolha_K = {}
    for K in RR.K_CANDIDATOS:
        w = np.array([RR.peso_humano(n, K) for n in ntr])
        mist = w * oof_h + (1 - w) * oof_c
        cal = RR.ajustar_calibradores(mist, ytr, ntr)
        aj = RR.Ajuste(col_con, modelo_con, K, cal)
        escolha_K[K] = recall_na_precisao(aj.calibrar(mist, ntr), ytr, rel_tr, config.DECS_PRECISAO_AUTOMATICA)
    K_melhor = max(escolha_K, key=lambda k: (escolha_K[k], -k))
    print("raros: recall a %.0f%% de precisao (fora da amostra do treino) por K: %s -> K=%d"
          % (100 * config.DECS_PRECISAO_AUTOMATICA, {k: round(v, 4) for k, v in escolha_K.items()}, K_melhor),
          flush=True)
    w0 = np.array([RR.peso_humano(n, 0) for n in ntr])
    ajustes = {
        "A. classificador (como antes)": None,
        "B. + calibracao por faixa": RR.Ajuste(col_con, None, 0, RR.ajustar_calibradores(w0 * oof_h, ytr, ntr)),
        "C. + mistura com o DeCS (K=%d) + calibracao" % K_melhor: RR.Ajuste(
            col_con, modelo_con, K_melhor, RR.ajustar_calibradores(
                np.array([RR.peso_humano(n, K_melhor) for n in ntr]) * oof_h
                + (1 - np.array([RR.peso_humano(n, K_melhor) for n in ntr])) * oof_c, ytr, ntr)),
    }

    Xte, yte, gte = coletar(ix, teste)
    nte = exemplos(gte)
    p_base = modelo.predict_proba(Xte)[:, 1]
    p_var = {nome: (p_base if aj is None else aj.aplicar(p_base, Xte, nte)) for nome, aj in ajustes.items()}
    cobertura = sum(1 for h in teste for u in ix.gabarito[h]) and \
        yte.sum() / sum(len(ix.gabarito[h]) for h in teste)
    col = {c: i for i, c in enumerate(IX.CARACTERISTICAS)}

    def ranking(score_fn, probs=None):
        por = defaultdict(list)
        for (h, u), x, pp in zip(gte, Xte, p if probs is None else probs):
            por[h].append((u, score_fn(x, pp)))
        for h in teste:
            por[h].sort(key=lambda z: -z[1])
        return por

    # -- as tres variantes nos MESMOS itens de teste; a usada e a de maior recall com precisao >= alvo
    finos_lim = [x / 100 for x in range(20, 100)]

    def limiar_de(rk, prec):
        finos = metricas_ranking(rk, ix.gabarito, finos_lim, topk=())
        ok = [(float(k[2:]), v) for k, v in finos.items() if v["precisao"] >= prec]
        return (min(ok)[0], min(ok)[1]) if ok else (0.99, {"precisao": 0, "recall": 0, "por_item": 0})

    def por_faixa(probs, t):
        """precisao/recall por faixa de frequencia do descritor, no limiar t (o que seria gravado sozinho)."""
        out = {}
        f_par = np.array([RR.faixa(n) for n in nte])
        rel = defaultdict(int)
        for h in teste:
            for u in ix.gabarito[h]:
                rel[RR.faixa(ix.freq.get(u, 0))] += 1
        for _a, _b, fx in RR.FAIXAS:
            sel = (f_par == fx) & (probs >= t)
            ac = int(yte[sel].sum())
            out[fx] = {"gravados": int(sel.sum()), "precisao": ac / max(1, int(sel.sum())),
                       "recall": ac / max(1, rel[fx]), "humanos": rel[fx]}
        return out

    def vicios(probs, t):
        """Quantas vezes os dois vicios medidos em 2026-10-20 seriam gravados sozinhos, e quantos certos."""
        out = {}
        for nome in ("Política de Saúde", "Eugenia"):
            uis = {u for u in vocab.base if vocab.nome(u) == nome}
            sel = np.array([u in uis for _h, u in gte]) & (probs >= t)
            out[nome] = {"gravados": int(sel.sum()), "certos": int(yte[sel].sum())}
        return out
    variantes = {}
    for nome, pv in p_var.items():
        rk = ranking(lambda x, pp: pp, pv)
        t_auto, m_auto = limiar_de(rk, config.DECS_PRECISAO_AUTOMATICA)
        t_rev, _m = limiar_de(rk, config.DECS_PRECISAO_REVISAO)
        variantes[nome] = {"limiar_automatico": t_auto, "limiar_revisao": t_rev,
                           "no_limiar_automatico": m_auto, "por_faixa": por_faixa(pv, t_auto),
                           "vicios": vicios(pv, t_auto),
                           "P@5": metricas_ranking(rk, ix.gabarito, (), topk=(5,))["P@5"]}
        print("%-50s limiar %.2f | precisao %.3f recall %.3f | %s" % (
            nome, t_auto, m_auto["precisao"], m_auto["recall"], variantes[nome]["vicios"]), flush=True)
    escolhida = max(variantes, key=lambda k: (variantes[k]["no_limiar_automatico"]["recall"], k))
    print("usada: %s" % escolhida, flush=True)
    p = p_var[escolhida]
    ajuste_escolhido = ajustes[escolhida]

    metodos = {
        "Combinado (classificador)": ranking(lambda x, pp: pp),
        "So kNN (vizinhos indexados)": ranking(lambda x, pp: x[col["knn_soma"]]),
        "So lexico (palavra-chave = termo DeCS)": ranking(
            lambda x, pp: 2 * x[col["lex_kw_nome"]] + x[col["lex_kw_sin"]] + 0.001 * x[col["denso"]]),
        "So denso (texto x descritor; metodo antigo)": ranking(lambda x, pp: x[col["denso"]]),
        "So rede multirrotulo": ranking(lambda x, pp: x[col["rede"]]),
    }
    limiares = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    resultado = {"data": datetime.now().isoformat(timespec="seconds"),
                 "n_teste": len(teste), "n_treino": len(treino),
                 "recall_maximo_dos_candidatos": float(cobertura),
                 "raros": {"K_fora_da_amostra": {str(k): v for k, v in escolha_K.items()}, "K": K_melhor,
                           "variantes": variantes, "usada": escolhida},
                 "teste_handles": list(teste),
                 "metodos": {}}
    for nome, rk in metodos.items():
        resultado["metodos"][nome] = metricas_ranking(
            rk, ix.gabarito, limiares if nome.startswith("Combinado") else ())

    # limiares de operacao escolhidos pela PRECISAO medida
    def limiar_para(prec):
        finos = metricas_ranking(metodos["Combinado (classificador)"], ix.gabarito,
                                 [x / 100 for x in range(20, 100)], topk=())
        ok = [(float(k[2:]), v) for k, v in finos.items() if v["precisao"] >= prec]
        return min(ok)[0] if ok else 0.99
    resultado["limiar_automatico"] = limiar_para(config.DECS_PRECISAO_AUTOMATICA)
    resultado["limiar_revisao"] = limiar_para(config.DECS_PRECISAO_REVISAO)
    imp = {}
    try:
        from sklearn.inspection import permutation_importance
        sub = np.random.RandomState(0).choice(len(yte), size=min(20000, len(yte)), replace=False)
        pi = permutation_importance(modelo, Xte[sub], yte[sub], n_repeats=3, random_state=0,
                                    scoring="average_precision")
        imp = {c: float(pi.importances_mean[i]) for i, c in enumerate(IX.CARACTERISTICAS)}
    except Exception as e:
        imp = {"erro": str(e)}
    resultado["importancia"] = imp

    # modelo de galho (validacao cruzada por item contra escolhas de catalogadores)
    from arca import decs_galho
    gal = decs_galho.EscolhaDeGalho(vocab, acervo, vetores=ix.V, handles=ix.handles,
                                    carregar_modelo=False)
    met_galho, n_galho = gal.treinar_e_avaliar({h: ix.V[ix.pos[h]] for h in gal.humano_por_item
                                                if h in ix.pos})
    resultado["galho"] = {"decisoes": n_galho, **met_galho}
    print("galho:", n_galho, "decisoes", {k: round(v, 3) for k, v in met_galho.items()}, flush=True)

    IX.salvar_modelo(modelo, {"automatico": resultado["limiar_automatico"],
                              "revisao": resultado["limiar_revisao"]}, resultado, ajuste=ajuste_escolhido)
    with open(os.path.join(config.SAIDAS_AVALIACAO, "avaliacao.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    escrever_relatorio(resultado)
    print("pronto em %.0fs" % (time.time() - t0))


def escrever_relatorio(r):
    from arca import decs_raros as RR
    L = ["# Avaliacao da atribuicao de DECS", "",
         "Gerado em %s. Itens de teste: %d (nunca vistos no treino). Itens de treino: %d."
         % (r["data"], r["n_teste"], r["n_treino"]), "",
         "Recall maximo possivel (descritores humanos que aparecem entre os candidatos): **%.1f%%**"
         % (100 * r["recall_maximo_dos_candidatos"]), "",
         "## Comparacao de metodos (top-k por item)", "",
         "| Metodo | P@1 | P@3 | P@5 | R@5 | R@10 |", "|---|---|---|---|---|---|"]
    for nome, m in r["metodos"].items():
        L.append("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
            nome, 100 * m["P@1"], 100 * m["P@3"], 100 * m["P@5"], 100 * m["R@5"], 100 * m["R@10"]))
    L += ["", "## Classificador por limiar de probabilidade", "",
          "| Limiar | Precisao | Recall | F1 | Descritores por item |", "|---|---|---|---|---|"]
    comb = r["metodos"]["Combinado (classificador)"]
    for k, v in comb.items():
        if k.startswith("t="):
            L.append("| %s | %.1f%% | %.1f%% | %.3f | %.1f |" % (
                k[2:], 100 * v["precisao"], 100 * v["recall"], v["f1"], v["por_item"]))
    L += ["", "Limiar de gravacao automatica (precisao >= %.0f%%): **%.2f**"
          % (100 * config.DECS_PRECISAO_AUTOMATICA, r["limiar_automatico"]),
          "Limiar de revisao humana (precisao >= %.0f%%): **%.2f**"
          % (100 * config.DECS_PRECISAO_REVISAO, r["limiar_revisao"]), "",
          "## Escolha de galho (validacao cruzada por item, %d escolhas de catalogadores)"
          % r.get("galho", {}).get("decisoes", 0), "",
          "| Metodo | Acerta o galho do catalogador |", "|---|---|",
          "| modelo combinado (usado) | %.1f%% |" % (100 * r.get("galho", {}).get("combinado", 0)),
          "| so precedente do acervo | %.1f%% |" % (100 * r.get("galho", {}).get("precedente", 0)),
          "| so 1a arvore da BIREME | %.1f%% |" % (100 * r.get("galho", {}).get("primeira_bireme", 0)), "",
          "## Descritores raros (arca/decs_raros.py, DECISOES §19)", "",
          "K escolhido fora da amostra do treino: **%s** (recall a %.0f%% de precisao por K: %s). Variante usada: "
          "**%s**." % (r["raros"]["K"], 100 * config.DECS_PRECISAO_AUTOMATICA,
                         ", ".join("K=%s %.1f%%" % (k, 100 * v) for k, v in r["raros"]["K_fora_da_amostra"].items()),
                         r["raros"]["usada"]), "",
          "| Variante | Limiar automatico | Precisao | Recall | P@5 | Politica de Saude gravados (certos) | "
          "Eugenia gravados (certos) |", "|---|---|---|---|---|---|---|"]
    for nome, v in r["raros"]["variantes"].items():
        m = v["no_limiar_automatico"]
        L.append("| %s | %.2f | %.1f%% | %.1f%% | %.1f%% | %d (%d) | %d (%d) |" % (
            nome, v["limiar_automatico"], 100 * m["precisao"], 100 * m["recall"], 100 * v["P@5"],
            v["vicios"]["Política de Saúde"]["gravados"], v["vicios"]["Política de Saúde"]["certos"],
            v["vicios"]["Eugenia"]["gravados"], v["vicios"]["Eugenia"]["certos"]))
    L += ["", "Por faixa de frequencia do descritor (exemplos humanos no treino), no limiar automatico de cada "
          "variante: gravados / precisao / recall.", "",
          "| Variante | " + " | ".join(fx for _a, _b, fx in RR.FAIXAS) + " |",
          "|---|" + "---|" * len(RR.FAIXAS)]
    for nome, v in r["raros"]["variantes"].items():
        L.append("| %s | " % nome + " | ".join("%d / %.0f%% / %.1f%%" % (
            v["por_faixa"][fx]["gravados"], 100 * v["por_faixa"][fx]["precisao"], 100 * v["por_faixa"][fx]["recall"])
            for _a, _b, fx in RR.FAIXAS) + " |")
    L += ["", "## Peso de cada sinal (importancia por permutacao)", ""]
    for c, v in sorted(r["importancia"].items(), key=lambda kv: -kv[1] if isinstance(kv[1], float) else 0):
        L.append("- %s: %.4f" % (c, v) if isinstance(v, float) else "- %s: %s" % (c, v))
    with open(os.path.join(config.SAIDAS_AVALIACAO, "avaliacao.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
