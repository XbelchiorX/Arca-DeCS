# -*- coding: utf-8 -*-
"""Acervo do passado: nos itens SEM nenhum DeCS, grava so as sugestoes muito recomendadas (ADICIONAR: limiar de
gravacao automatica, precisao >= 90% em saidas/avaliacao/avaliacao.md, e as palavras-chave do autor que ja sao
descritores DeCS). As apenas possiveis nao entram.
[DECS-ORIGINAL]/[DECS-AUTO] e a releitura ao vivo ficam no gravar_novos (regras 3, 4 e 7).
Uso: python ferramentas/decs/sugerir_sem_decs.py            (simulacao: conta o que gravaria)
     python ferramentas/decs/sugerir_sem_decs.py --gravar
"""
import argparse
import os
import sys
from collections import Counter

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gravar", action="store_true")
    a = ap.parse_args()
    from arca import decs_correcao as C, decs_menu as M, idiomas as I, preparo
    if preparo.garantir():
        sys.exit("dados locais do DeCS faltando")
    itens = I.mapear_objetos(I.criar_cliente(), 100, 8)
    # sem DeCS, ou com DeCS so do modelo ([DECS-ORIGINAL] vazio): esses recebem o que faltar (ex.: palavras-chave
    # do autor que sao DeCS, regra de 2026-10-20), sem tocar no que ja tem
    sem = [it for it in itens if it.get("handle") and (not (it.get("metadata") or {}).get(config.CAMPO_DECS)
                                                      or R.ler_original(it.get("metadata") or {}) == [])]
    print("%d itens mapeados; %d sem DeCS humano" % (len(itens), len(sem)), flush=True)
    ctx, _ = M.contexto(sugerir=True)
    if not ctx.sugerir:
        sys.exit("modelo de sugestao nao carregado")
    from concurrent.futures import ThreadPoolExecutor
    cont, n, futs = Counter(), 0, []

    def gravar(h, fortes, tem):
        try:
            return C.gravar_novos(I.get_thread_client()._c, h, fortes, ctx, acrescentar=tem).split()[0]
        except Exception as e:
            print("  ERRO %s: %s" % (h, str(e)[:200]), flush=True)
            return "erro"

    with ThreadPoolExecutor(8) as ex:          # o plano e local (rapido); a gravacao (PATCH + releitura) em paralelo
        for k, it in enumerate(sem, 1):
            try:
                linhas, _f, _k = C.planejar(it["handle"], C.item_do_dspace(it), ctx)
            except Exception as e:
                cont["erro"] += 1
                print("  ERRO %s: %s" % (it["handle"], str(e)[:200]), flush=True)
                continue
            # so as palavras-chave do autor: o classificador ficou desligado na gravacao sem revisao depois da
            # amostra de 2026-10-20 (Politica de Saude em 84% e "Eugenia" p=0,999 sem relacao; DECISOES §18)
            fortes = [l for l in linhas if l["acao"] == "ADICIONAR" and l["metodo"] == "palavra_chave_autor"]
            if not fortes:
                cont["sem_sugestao_forte"] += 1
            elif a.gravar:
                tem = bool((it.get("metadata") or {}).get(config.CAMPO_DECS))
                futs.append(ex.submit(gravar, it["handle"], fortes, tem))
                n += len(fortes)
            else:
                cont["gravaria"] += 1
                n += len(fortes)
            if k % 200 == 0:
                feitos = [f.result() for f in futs if f.done()]
                print("%d/%d %s gravados=%s descritores=%d" % (k, len(sem), dict(cont), dict(Counter(feitos)), n),
                      flush=True)
    for f in futs:
        cont[f.result()] += 1
    print("FIM itens: %s | descritores %s: %d" % (dict(cont), "gravados" if a.gravar else "a gravar", n))


if __name__ == "__main__":
    main()
