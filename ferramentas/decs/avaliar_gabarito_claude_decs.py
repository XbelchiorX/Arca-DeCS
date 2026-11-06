# -*- coding: utf-8 -*-
"""Motores de DeCS (A embedding x B IA) contra a indexacao CEGA do Claude, a pedido do responsavel (2026-10-17).

O Claude indexou 100 dos 500 itens do teste (sorteio, semente 20261017) lendo so titulo, palavras-chave e resumo,
sem ver o DeCS do catalogador nem as saidas dos motores: dados/gabarito_claude_decs/rotulos_claude.txt (termos) e
galhos_claude.txt (galho de cada descritor com mais de um caminho). E um gabarito de AVALIACAO: nunca entra no
treino (regra 7 do CLAUDE.md).

Le as saidas ja gravadas em saidas/avaliacao/comparacao_motores_decs.json (nada e rodado de novo). O galho do motor A
sai do modelo de galho do Qwen3-8B (DADOS_LOCAIS/modelos_qwen3-8b), com os vetores do cache.
Saida: saidas/avaliacao/gabarito_claude_decs.md e .json
Uso: python ferramentas/decs/avaliar_gabarito_claude_decs.py
"""
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "ferramentas", "decs"))
os.environ.setdefault("DECS_MODELO_EMBEDDING", "qwen3-8b")
import torch  # noqa: E402

from arca import config, decs_acervo, decs_galho, decs_indexador as IX, decs_vocabulario as V  # noqa: E402
from treinar_avaliar import selecionar_amostra  # noqa: E402

PASTA = os.path.join(RAIZ, "dados", "gabarito_claude_decs")
AV = config.SAIDAS_AVALIACAO


def ler_rotulos(vocab):
    gab = {}
    for linha in open(os.path.join(PASTA, "rotulos_claude.txt"), encoding="utf-8"):
        if not linha.startswith("icict"):
            continue
        h, termos = linha.split("|", 1)
        uis = []
        for t in (x.strip() for x in termos.split(";")):
            if not t:
                continue
            u = [x for x in vocab.uis_do_termo(t)[0] if vocab.caminhos(x)]
            if not u:
                sys.exit("termo sem descritor gravavel: %s (%s)" % (t, h))
            uis.append(u[0])
        gab[h.strip()] = list(dict.fromkeys(uis))
    return gab


def ler_galhos(vocab, gab):
    geral, por_item = {}, {}
    for linha in open(os.path.join(PASTA, "galhos_claude.txt"), encoding="utf-8"):
        p = linha.split()
        if not p or p[0].startswith("#"):
            continue
        (por_item.setdefault(p[2], {}) if len(p) > 2 else geral)[p[0]] = int(p[1])
    cam = {}
    for h, uis in gab.items():
        for u in uis:
            cs = vocab.caminhos(u)
            if len(cs) > 1:
                k = por_item.get(h, {}).get(u, geral.get(u))
                if k is None:
                    sys.exit("sem galho escolhido: %s %s" % (u, h))
                cam[(h, u)] = cs[k - 1]
    return cam


def metricas(gab, pred):
    ac = sug = ref = 0
    for h, g in gab.items():
        p = set(pred.get(h, []))
        ac += len(p & set(g))
        sug += len(p)
        ref += len(g)
    pr = ac / sug if sug else 0.0
    rc = ac / ref if ref else 0.0
    return {"precisao": round(pr, 3), "recall": round(rc, 3),
            "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0,
            "por_item": round(sug / len(gab), 2), "acertos": ac, "sugeridos": sug}


def main():
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    gab = ler_rotulos(vocab)
    cam_claude = ler_galhos(vocab, gab)
    res = json.load(open(os.path.join(AV, "comparacao_motores_decs.json"), encoding="utf-8"))["itens"]
    hs = [h for h in gab if h in res]
    gab = {h: gab[h] for h in hs}
    A = {h: res[h]["A"] for h in hs}
    B = {h: [d["ui"] for d in res[h]["B"]] for h in hs}
    B_cam = {h: {d["ui"]: d["caminho"] for d in res[h]["B"]} for h in hs}
    humano = {h: res[h]["humano"] for h in hs}
    sem_b = [h for h in hs if not res[h]["B"] and not res[h].get("resposta_llm")]

    # galho do motor A: o mesmo modelo de galho do teste (Qwen3-8B), com os vetores do cache
    decs_galho.ARQ_MODELO_GALHO = os.path.join(config.DADOS_LOCAIS, "modelos_qwen3-8b", "modelo_galho.pkl")
    acervo, _ = selecionar_amostra(decs_acervo.carregar_acervo(), vocab, 500, 4000, 8000)
    c = torch.load(IX.ARQ_EMB_ITENS, weights_only=False)
    pos = {k: i for i, k in enumerate(c["chaves"])}
    handles = [h for h in acervo if h in pos]
    vetores = c["vetores"][[pos[h] for h in handles]]
    galho = decs_galho.EscolhaDeGalho(vocab, acervo, vetores=vetores, handles=handles)
    if galho.modelo is None:
        sys.exit("modelo de galho do 8B nao encontrado em %s" % decs_galho.ARQ_MODELO_GALHO)
    vet = {h: vetores[i] for i, h in enumerate(handles)}

    gal = {"A_modelo_de_galho": [0, 0], "B_llm": [0, 0], "B_com_modelo_de_galho": [0, 0]}
    exemplos = []
    for h in hs:
        for nome, uis, cam_de in (("A_modelo_de_galho", A[h], None), ("B_llm", B[h], B_cam[h]),
                                  ("B_com_modelo_de_galho", B[h], None)):
            ctx = galho.contexto(h, [vocab.nome(u) for u in uis], vet.get(h))
            for u in uis:
                if (h, u) not in cam_claude:
                    continue
                cm = cam_de[u] if cam_de else galho.escolher(u, excluir_handle=h, contexto=ctx)[0]
                gal[nome][1] += 1
                gal[nome][0] += cm == cam_claude[(h, u)]
                if nome != "B_com_modelo_de_galho" and len(exemplos) < 400:
                    exemplos.append({"item": h, "motor": nome[0], "descritor": vocab.nome(u),
                                     "claude": cam_claude[(h, u)], "motor_galho": cm})

    r = {"itens": len(hs), "termos_claude": sum(len(g) for g in gab.values()),
         "A_ate_revisao": metricas(gab, A), "B_llm": metricas(gab, B), "catalogador": metricas(gab, humano),
         "galho": {k: {"acertos": a, "casos": n, "taxa": round(a / n, 3) if n else None}
                   for k, (a, n) in gal.items()},
         "sem_resposta_B": sem_b}
    os.makedirs(AV, exist_ok=True)
    json.dump({"resumo": r, "gabarito": {h: [[u, vocab.nome(u)] for u in gab[h]] for h in hs},
               "galho_claude": {"%s|%s" % k: v for k, v in cam_claude.items()}, "exemplos_galho": exemplos},
              open(os.path.join(AV, "gabarito_claude_decs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    L = ["# Motores de DeCS contra a indexacao cega do Claude", "",
         "%d itens (sorteio entre os 500 ultimos depositados com DeCS). O Claude leu so titulo, palavras-chave e "
         "resumo, sem o DeCS do catalogador e sem as saidas dos motores: %d termos (%.1f por item)."
         % (len(hs), r["termos_claude"], r["termos_claude"] / len(hs)), "",
         "| Quem | Precisao | Recall | F1 | Descritores por item |", "|---|---|---|---|---|"]
    for nome, k in (("A - embedding (inclui revisao)", "A_ate_revisao"), ("B - IA (Qwen3.5-9B)", "B_llm"),
                    ("Catalogador (referencia)", "catalogador")):
        m = r[k]
        L.append("| %s | %.1f%% | %.1f%% | %.3f | %.1f |" % (nome, 100 * m["precisao"], 100 * m["recall"],
                                                            m["f1"], m["por_item"]))
    L += ["", "Galho (descritores com mais de um caminho que o Claude tambem escolheu):", ""]
    for k, v in r["galho"].items():
        L.append("- %s: %d de %d (%s)" % (k, v["acertos"], v["casos"],
                                          "%.1f%%" % (100 * v["taxa"]) if v["taxa"] is not None else "-"))
    if sem_b:
        L += ["", "B sem resposta (texto maior que a janela do modelo): " + ", ".join(sem_b)]
    open(os.path.join(AV, "gabarito_claude_decs.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
