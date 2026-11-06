# -*- coding: utf-8 -*-
"""Mede a segunda opiniao do LLM (arca/decs_segunda_opiniao.py) na faixa de revisao do DeCS.

Usa os itens de TESTE do ultimo ferramentas/decs/treinar_avaliar.py (mesma semente: nunca vistos no treino) e o
modelo salvo. Para cada item, os candidatos com probabilidade entre o limiar de revisao e o automatico sao
julgados pelo LLM; compara com o que o catalogador atribuiu:
  - precisao da faixa sem o LLM (quanto da fila de revisao o catalogador teria aceitado);
  - precisao do que o LLM confirma, e quanto dos acertos da faixa ele preserva (recall na faixa);
  - tamanho da fila antes e depois.
Precisa da GPU livre (sobe o servidor do professor). Uso: python ferramentas/decs/avaliar_segunda_opiniao_decs.py [N=300]
"""
import json
import os
import random
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "ferramentas", "decs"))
from arca import config, decs_acervo, decs_indexador as IX, decs_segunda_opiniao as SO  # noqa: E402
from arca import decs_vocabulario as V, ods_professor as P  # noqa: E402
from treinar_avaliar import duplicatas  # noqa: E402


def main(n):
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    acervo = decs_acervo.carregar_acervo()
    ix = IX.montar_indexador(vocab, acervo, com_modelo=True)
    if not ix.modelo or not isinstance(ix.limiar, dict):
        sys.exit("Modelo de DECS nao treinado: rode ferramentas/decs/treinar_avaliar.py antes.")
    lim_rev, lim_auto = ix.limiar["revisao"], ix.limiar["automatico"]
    elegiveis = sorted(h for h in ix.handles if ix.gabarito.get(h) and decs_acervo.tem_texto_suficiente(acervo[h]))
    random.seed(20261002)
    random.shuffle(elegiveis)
    teste = elegiveis[:1500][:n]                     # mesmos itens de teste da avaliacao
    ix.recalcular_prior(excluir=elegiveis[:1500])
    P.iniciar_servidor()
    t0 = time.time()
    fila = acerto_fila = conf = acerto_conf = falhas = 0
    detalhes = []
    for i, h in enumerate(teste, 1):
        vet = ix.V[ix.pos[h]]
        pts = ix.pontuar(acervo[h], vet, excluir=duplicatas(ix, h, vet), proprio=h)
        faixa = [(u, p) for u, p in pts if lim_rev <= p < lim_auto]
        humano = set(ix.gabarito[h])
        r = SO.confirmar(acervo[h], faixa, vocab)
        if r is None:
            falhas += 1
            r = [u for u, _p in faixa]
        fila += len(faixa)
        acerto_fila += sum(u in humano for u, _p in faixa)
        conf += len(r)
        acerto_conf += sum(u in humano for u in r)
        detalhes.append({"handle": h, "faixa": [[u, round(p, 3), u in humano] for u, p in faixa], "llm": r})
        if i % 25 == 0:
            print("  %d/%d (%.1f s/item)" % (i, len(teste), (time.time() - t0) / i), flush=True)
    P.parar_servidor()
    res = {"professor": P.NOME, "itens": len(teste), "limiar_revisao": lim_rev, "limiar_automatico": lim_auto,
           "fila_sem_llm": fila, "precisao_fila_sem_llm": round(acerto_fila / max(fila, 1), 3),
           "fila_com_llm": conf, "precisao_llm": round(acerto_conf / max(conf, 1), 3),
           "acertos_preservados": round(acerto_conf / max(acerto_fila, 1), 3), "falhas_json": falhas}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    saida = os.path.join(config.SAIDAS, "avaliacao", "segunda_opiniao_decs.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump({"resumo": res, "itens": detalhes}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
