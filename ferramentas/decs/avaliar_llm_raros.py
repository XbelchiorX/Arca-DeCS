# -*- coding: utf-8 -*-
"""Mede o passo 3 dos descritores raros (arca/decs_raros.py, DECISOES §19): o LLM confirmar, antes da gravacao
automatica, os descritores com menos de decs_raros.K_LLM exemplos humanos.

Itens: os de TESTE do ultimo ferramentas/decs/treinar_avaliar.py (gravados em saidas/avaliacao/avaliacao.json,
nunca vistos no treino), com o modelo salvo (inclusive a mistura/calibracao escolhida). Para cada item, os
candidatos raros com probabilidade >= limiar automatico (os que seriam gravados sozinhos) vao para o LLM; compara
com o catalogador:
  - precisao dos raros gravados sem o LLM e com o LLM (so os confirmados);
  - quanto dos acertos o LLM preserva (recall relativo).
Precisa da GPU livre (sobe o servidor do professor; ODS_PROFESSOR escolhe o modelo).
Uso: python ferramentas/decs/avaliar_llm_raros.py [N=1500]
"""
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "ferramentas", "decs"))
from arca import config, decs_acervo, decs_indexador as IX, decs_raros as RR, decs_segunda_opiniao as SO  # noqa: E402
from arca import decs_vocabulario as V, ods_professor as P  # noqa: E402
from treinar_avaliar import duplicatas  # noqa: E402


def main(n):
    aval = json.load(open(os.path.join(config.SAIDAS_AVALIACAO, "avaliacao.json"), encoding="utf-8"))
    teste = aval.get("teste_handles")
    if not teste:
        sys.exit("avaliacao.json sem teste_handles: rode ferramentas/decs/treinar_avaliar.py de novo.")
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    acervo = decs_acervo.carregar_acervo()
    ix = IX.montar_indexador(vocab, acervo, com_modelo=True)
    ix.recalcular_prior(excluir=teste)                 # n e prior sem os itens de teste, como na avaliacao
    lim = ix.limiar["automatico"]
    P.iniciar_servidor()
    t0 = time.time()
    tot = ac = conf = ac_conf = falhas = itens_com = 0
    det = []
    for i, h in enumerate(teste[:n], 1):
        vet = ix.V[ix.pos[h]]
        pts = ix.pontuar(acervo[h], vet, excluir=duplicatas(ix, h, vet), proprio=h)
        raros = [(u, p) for u, p in pts if p >= lim and ix.freq.get(u, 0) < RR.K_LLM]
        if not raros:
            continue
        itens_com += 1
        humano = set(ix.gabarito[h])
        r = SO.confirmar(acervo[h], raros, vocab)
        if r is None:
            falhas += 1
            r = []
        tot += len(raros)
        ac += sum(u in humano for u, _p in raros)
        conf += len(r)
        ac_conf += sum(u in humano for u in r)
        det.append({"handle": h, "raros": [[vocab.nome(u), round(p, 3), ix.freq.get(u, 0), u in humano, u in r]
                                           for u, p in raros]})
        if itens_com % 20 == 0:
            print("  %d/%d itens; %d com raros (%.1f s/item com raros)" % (i, n, itens_com,
                  (time.time() - t0) / itens_com), flush=True)
    P.parar_servidor()
    res = {"professor": P.NOME, "itens_teste": min(n, len(teste)), "itens_com_raros": itens_com, "K_LLM": RR.K_LLM,
           "limiar_automatico": lim, "raros_gravados_sem_llm": tot, "precisao_sem_llm": round(ac / max(tot, 1), 3),
           "raros_gravados_com_llm": conf, "precisao_com_llm": round(ac_conf / max(conf, 1), 3),
           "acertos_preservados": round(ac_conf / max(ac, 1), 3), "falhas_json": falhas}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with open(os.path.join(config.SAIDAS_AVALIACAO, "llm_raros_decs.json"), "w", encoding="utf-8") as f:
        json.dump({"resumo": res, "itens": det}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
