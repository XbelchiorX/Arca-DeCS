# -*- coding: utf-8 -*-
"""Mede se o CONTEUDO do documento (titulo + palavras-chave + resumo, via embedding) melhora a escolha do galho
(pedido do responsavel, 2026-10-17). So medicao: nada e gravado no ARCA nem no modelo em uso.

Referencia: os galhos que os CATALOGADORES gravaram (decisao do responsavel). Treino e teste so com esses itens;
validacao cruzada por item (5 dobras): o proprio item nunca entra nas contas dele.
Tres versoes, contra o catalogador:
  1. sinais atuais (costume do acervo, coerencia com os outros descritores do item, vizinhos) = modelo de hoje;
  2. so conteudo: semelhanca do documento com os documentos que os catalogadores puseram em cada galho
     (do mesmo descritor e da mesma categoria) e voto dos documentos mais parecidos que tem esse descritor;
  3. tudo combinado.
Tambem: acerto por faixa de confianca (para decidir onde gravar direto).
Uso: python ferramentas/decs/medir_galho_conteudo.py [--modelo e5-large|qwen3-8b]
Saida: saidas/avaliacao/galho_conteudo_<modelo>.md e .json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="e5-large")
    a = ap.parse_args()
    os.environ["DECS_MODELO_EMBEDDING"] = a.modelo
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from arca import config, decs_acervo, decs_galho, decs_indexador as IX, decs_vocabulario as V
    from arca.decs_indexador import valores_humanos
    t0 = time.time()
    vocab = V.carregar_vocabulario()
    acervo = decs_acervo.carregar_acervo()
    c = torch.load(IX.ARQ_EMB_ITENS, weights_only=False)
    pos = {k: i for i, k in enumerate(c["chaves"])}
    handles = [h for h in acervo if h in pos]
    vet = torch.nn.functional.normalize(c["vetores"][[pos[h] for h in handles]].float(), dim=1)
    vpos = {h: i for i, h in enumerate(handles)}
    galho = decs_galho.EscolhaDeGalho(vocab, acervo, vetores=vet, handles=handles, carregar_modelo=False)
    print("itens com vetor: %d de %d; itens com galho humano: %d (%.0fs)"
          % (len(handles), len(acervo), len(galho.humano_por_item), time.time() - t0), flush=True)

    # escolhas humanas (item, descritor, galho) com mais de um galho possivel e vetor disponivel
    esc = [(h, u, cam) for h, m in galho.humano_por_item.items() if h in vpos
           for u, cam in m.items() if cam in vocab.caminhos(u)]
    cat = lambda cm: cm.split("::")[0]
    dim = vet.shape[1]
    S_path, N_path = defaultdict(lambda: torch.zeros(dim)), Counter()
    S_cat, N_cat = defaultdict(lambda: torch.zeros(dim)), Counter()
    por_ui = defaultdict(list)
    for h, u, cam in esc:
        v = vet[vpos[h]]
        S_path[(u, cam)] += v
        N_path[(u, cam)] += 1
        S_cat[cat(cam)] += v
        N_cat[cat(cam)] += 1
        por_ui[u].append((h, cam))
    proprio_cat = defaultdict(Counter)
    for h, u, cam in esc:
        proprio_cat[h][cat(cam)] += 1

    def cos_centro(S, n, v, menos):
        if n - menos <= 0:
            return 0.0, 0
        cvec = (S - v * menos) / (n - menos)
        nn = float(cvec.norm())
        return (float(cvec @ v) / nn if nn else 0.0), n - menos

    linhas_X, ys, grupos, dobra_item = [], [], [], []
    ordem_item = {h: i for i, h in enumerate(sorted({h for h, _u, _c in esc}))}
    for gi, (h, u, escolhido) in enumerate(esc):
        v = vet[vpos[h]]
        cams = vocab.caminhos(u)
        ctx = galho.contexto(h, valores_humanos(acervo[h]), v)
        Xold = galho.caracteristicas(u, cams, h, ctx, escolhido=escolhido)
        # voto dos documentos mais parecidos que tem ESTE descritor com galho humano (sem o proprio)
        outros = [(hh, cc) for hh, cc in por_ui[u] if hh != h]
        votos = Counter()
        if outros:
            M = vet[[vpos[hh] for hh, _ in outros]]
            s = (M @ v).numpy()
            for j in np.argsort(-s)[:15]:
                votos[outros[j][1]] += max(float(s[j]), 0.0) ** 4
        tv = sum(votos.values())
        for k, cm in enumerate(cams):
            menos = 1 if cm == escolhido else 0
            cp, npth = cos_centro(S_path[(u, cm)], N_path[(u, cm)], v, menos)
            cc, _ = cos_centro(S_cat[cat(cm)], N_cat[cat(cm)], v, proprio_cat[h][cat(cm)])
            linhas_X.append(list(Xold[k]) + [cp, np.log1p(npth), cc, votos.get(cm, 0.0) / tv if tv else 0.0])
            ys.append(1 if cm == escolhido else 0)
            grupos.append(gi)
            dobra_item.append(ordem_item[h] % 5)
    X, y, g, dob = np.array(linhas_X, dtype=np.float32), np.array(ys), np.array(grupos), np.array(dobra_item)
    n_old = len(decs_galho.NOMES)
    versoes = {"1_sinais_atuais": list(range(n_old)),
               "2_so_conteudo": list(range(n_old, X.shape[1])),
               "3_combinado": list(range(X.shape[1]))}
    print("escolhas humanas multigalho: %d; linhas: %d (%.0fs)" % (len(esc), len(y), time.time() - t0), flush=True)

    res, confs = {}, {}
    for nome, cols in versoes.items():
        acertos, total = 0, 0
        conf = confs.setdefault(nome, [])
        for k in range(5):
            tr, te = dob != k, dob == k
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
            m.fit(X[tr][:, cols], y[tr])
            p = m.predict_proba(X[te][:, cols])[:, 1]
            por = defaultdict(list)
            for gg, pp, yy in zip(g[te], p, y[te]):
                por[gg].append((pp, yy))
            for cands in por.values():
                total += 1
                cands.sort(key=lambda t: -t[0])
                acertos += cands[0][1]
                if True:
                    soma = sum(t[0] for t in cands) or 1.0
                    conf.append((cands[0][0] / soma, cands[0][1]))
        res[nome] = round(acertos / total, 4)
        print("%s: %.1f%% de %d escolhas" % (nome, 100 * acertos / total, total), flush=True)

    faixas_por = {}
    for nome, conf in confs.items():
        conf.sort(key=lambda t: -t[0])
        faixas_por[nome] = []
        for lim in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.0):
            sel = [ok for cf, ok in conf if cf >= lim]
            if sel:
                faixas_por[nome].append({"confianca_min": lim, "cobertura": round(len(sel) / len(conf), 3),
                                         "acerto": round(sum(sel) / len(sel), 3)})
    faixas = faixas_por["1_sinais_atuais"]
    saida = {"modelo_embedding": a.modelo, "escolhas": len(esc), "itens": len(ordem_item),
             "acerto": res, "por_confianca": faixas_por}
    out = os.path.join(config.SAIDAS_AVALIACAO, "galho_conteudo_%s" % a.modelo)
    json.dump(saida, open(out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    L = ["# Galho pelo conteudo do documento (%s)" % a.modelo, "",
         "%d escolhas de galho feitas por catalogadores em %d itens (descritores com mais de um galho); validacao "
         "cruzada por item (5 dobras)." % (len(esc), len(ordem_item)), "",
         "| Versao | Acerto contra o catalogador |", "|---|---|"]
    L += ["| %s | %.1f%% |" % (k, 100 * v) for k, v in res.items()]
    L += ["", "Versao 1 (sinais atuais), por confianca:", "", "| Confianca minima | Cobertura | Acerto |", "|---|---|---|"]
    L += ["| %.2f | %.0f%% | %.1f%% |" % (f["confianca_min"], 100 * f["cobertura"], 100 * f["acerto"]) for f in faixas]
    open(out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("pronto em %.0f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
