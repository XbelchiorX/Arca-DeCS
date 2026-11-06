# -*- coding: utf-8 -*-
"""So leitura. Hipotese de 2026-10-21 (DECISOES §19): os vicios do classificador ("Politica de Saude", "Eugenia")
nao vem de o descritor ser raro, e sim de o ITEM ter pouco texto. A avaliacao mede itens com texto; a gravacao em
massa foi em itens sem DeCS, muitos com resumo curto ou sem resumo.

Mede, nos itens de TESTE do ultimo treinar_avaliar.py (nunca vistos) e com o modelo salvo, a precisao do que seria
gravado sozinho (p >= limiar automatico) por faixa do tamanho do resumo, contra o catalogador. Mede tambem a
populacao onde o modelo foi aplicado (itens SEM DeCS humano): a distribuicao do tamanho do resumo.
Uso: python ferramentas/decs/avaliar_por_texto.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "ferramentas", "decs"))
from arca import config, decs_acervo, decs_indexador as IX, decs_vocabulario as V  # noqa: E402
from treinar_avaliar import duplicatas  # noqa: E402

FAIXAS = [(0, 0, "sem resumo"), (1, 199, "1-199"), (200, 599, "200-599"), (600, 10 ** 9, "600+")]


def tam_resumo(item):
    return sum(len((x.get("v") or "").strip()) for x in item.get("resumos") or [])


def faixa(n):
    return next(nome for a, b, nome in FAIXAS if a <= n <= b)


def main():
    aval = json.load(open(os.path.join(config.SAIDAS_AVALIACAO, "avaliacao.json"), encoding="utf-8"))
    teste = aval["teste_handles"]
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    acervo = decs_acervo.carregar_acervo()
    ix = IX.montar_indexador(vocab, acervo, com_modelo=True)
    ix.recalcular_prior(excluir=teste)
    lim = ix.limiar["automatico"]
    vic = {u for u in vocab.base if vocab.nome(u) in ("Política de Saúde", "Eugenia")}
    m = defaultdict(lambda: Counter())
    for h in teste:
        it = acervo[h]
        fx = faixa(tam_resumo(it))
        vet = ix.V[ix.pos[h]]
        pts = ix.pontuar(it, vet, excluir=duplicatas(ix, h, vet), proprio=h)
        humano = set(ix.gabarito[h])
        m[fx]["itens"] += 1
        for u, p in pts:
            if p >= lim:
                m[fx]["gravados"] += 1
                m[fx]["certos"] += u in humano
                if u in vic:
                    m[fx]["vicios"] += 1
                    m[fx]["vicios_certos"] += u in humano
    pop = Counter(faixa(tam_resumo(r)) for h, r in acervo.items() if not IX.descritores_humanos(r, vocab))
    pop_teste = Counter(faixa(tam_resumo(acervo[h])) for h in teste)
    res = {"limiar_automatico": lim, "por_tamanho_do_resumo": {}, "populacao_sem_decs": dict(pop),
           "populacao_teste": dict(pop_teste)}
    print("limiar automatico %.2f" % lim)
    print("%-11s %6s %9s %9s %8s | %s | %s" % ("resumo", "itens", "gravados", "precisao", "vicios",
                                             "% dos itens de teste", "% dos itens SEM DeCS"))
    for _a, _b, fx in FAIXAS:
        c = m[fx]
        prec = c["certos"] / max(1, c["gravados"])
        res["por_tamanho_do_resumo"][fx] = dict(c, precisao=round(prec, 3))
        print("%-11s %6d %9d %8.1f%% %4d(%d) | %5.1f%% | %5.1f%%" % (
            fx, c["itens"], c["gravados"], 100 * prec, c["vicios"], c["vicios_certos"],
            100 * pop_teste[fx] / max(1, len(teste)), 100 * pop[fx] / max(1, sum(pop.values()))))
    with open(os.path.join(config.SAIDAS_AVALIACAO, "avaliacao_por_texto.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
