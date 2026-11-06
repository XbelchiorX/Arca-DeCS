# -*- coding: utf-8 -*-
"""Em qual galho da arvore um descritor deve ser gravado.

Metade dos descritores do DeCS esta em mais de um galho (ex.: Gestantes em
"Pessoas::Mulheres::Gestantes" e em "Estudos Populacionais em Saude
Publica::...::Pessoas::Gestantes"). O DeCS nao diz qual e o principal, mas o
DSpace grava UM caminho. Alguem tem de escolher.

Regra:
1. O catalogador ja escolheu um galho valido para este item -> esse.
2. MODELO DE GALHO (dados/modelo_galho.pkl, treinado na D5): combina, para
   cada galho candidato,
     - precedente: fracao das escolhas dos catalogadores do ARCA para este
       descritor que foram neste galho (sem contar o proprio item);
     - se e a 1a arvore da BIREME;
     - coerencia: quantos outros descritores do item estao na mesma
       categoria (1o nivel do caminho);
     - vizinhos: voto, ponderado pela similaridade, dos 50 documentos mais
       parecidos que tem este descritor com galho escolhido por catalogador.
3. Sem modelo: precedente do acervo -> 1a arvore da BIREME.

Medido contra 5.887 escolhas de galho de catalogadores (2026-10-02, validacao
cruzada por item; recalculado a cada D5 em saidas/avaliacao/avaliacao.md):
    combinado (modelo) ..................... 80,7%
    so vizinhos ............................ 71,9%
    so precedente do acervo ................ 71,7%
    so 1a arvore da BIREME ................. 65,8%
    so coerencia com o item ................ 45,1%
Cada sinal sozinho e fraco; o modelo aprende quando confiar em cada um
(pergunta do responsavel: "o que os catalogadores mais escolheram nem sempre
diz sobre o registro em questao").
"""
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch

from . import config
from .decs_indexador import valores_humanos
from .decs_interpretar import CAMINHO_ANTIGO, CAMINHO_VALIDO, interpretar
from .decs_vocabulario import norm

ARQ_MODELO_GALHO = os.path.join(config.DADOS_LOCAIS, "modelo_galho.pkl")
VIZINHOS = 50
NOMES = ["precedente_fracao", "precedente_log", "primeira_bireme", "coerencia",
         "vizinhos_fracao", "n_galhos"]


def _categoria(caminho):
    """1o nivel do valor gravavel (a categoria em que o galho cai)."""
    return norm((caminho or "").split("::")[0])


class EscolhaDeGalho:
    def __init__(self, vocab, acervo, vetores=None, handles=None, carregar_modelo=True):
        self.vocab = vocab
        self.acervo = acervo
        # handle -> {ui: caminho escolhido por humano}; ui -> Counter(caminho)
        self.humano_por_item = defaultdict(dict)
        self.uso = defaultdict(Counter)
        for h, item in acervo.items():
            for v in valores_humanos(item):
                r = interpretar(v, vocab)
                # CAMINHO_ANTIGO tambem e escolha humana de galho: e o mesmo no
                # escrito no formato anterior (ex.: sem a categoria de topo)
                if (r["tipo"] in (CAMINHO_VALIDO, CAMINHO_ANTIGO) and r["ui"]
                        and r["caminho_humano"] and len(vocab.caminhos(r["ui"])) > 1):
                    self.humano_por_item[h][r["ui"]] = r["caminho_humano"]
                    self.uso[r["ui"]][r["caminho_humano"]] += 1
        self.estatistica = Counter()
        # indice de vizinhos: so itens com alguma escolha humana de galho
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._viz_handles = []
        self._viz_V = None
        if vetores is not None and handles is not None:
            pos = {h: i for i, h in enumerate(handles)}
            self._viz_handles = [h for h in self.humano_por_item if h in pos]
            if self._viz_handles:
                self._viz_V = vetores[[pos[h] for h in self._viz_handles]].float().to(self.dev)
        self.modelo = None
        if carregar_modelo and os.path.exists(ARQ_MODELO_GALHO):
            with open(ARQ_MODELO_GALHO, "rb") as f:
                salvo = pickle.load(f)
            if salvo.get("nomes") == NOMES:
                self.modelo = salvo["modelo"]

    # -- contexto do item (calculado uma vez por item) -----------------------
    def vizinhos(self, vetor, excluir_handle=None):
        if self._viz_V is None or vetor is None:
            return []
        s = (self._viz_V @ vetor.to(self.dev).float().unsqueeze(1)).squeeze(1)
        top = torch.topk(s, k=min(VIZINHOS + 1, s.numel()))
        saida = [(self._viz_handles[j], v) for v, j in zip(top.values.tolist(), top.indices.tolist())
                 if self._viz_handles[j] != excluir_handle]
        return saida[:VIZINHOS]

    def categorias_do_item(self, valores):
        """Quantos descritores do item caem em cada categoria (1o nivel)."""
        c = Counter()
        for v in valores:
            r = interpretar(v, self.vocab)
            if not r["ui"]:
                continue
            if r["tipo"] in (CAMINHO_VALIDO, CAMINHO_ANTIGO) and r["caminho_humano"]:
                c[_categoria(r["caminho_humano"])] += 1
            else:
                cams = self.vocab.caminhos(r["ui"])
                for cam in cams:
                    c[_categoria(cam)] += 1.0 / len(cams)
        return c

    def contexto(self, handle, valores, vetor):
        return {"categorias": self.categorias_do_item(valores),
                "vizinhos": self.vizinhos(vetor, handle)}

    # -- caracteristicas de cada galho candidato -----------------------------
    def caracteristicas(self, ui, caminhos, excluir_handle, contexto, escolhido=None):
        """Uma linha por caminho candidato (mesma funcao no treino e no uso)."""
        uso = Counter(self.uso.get(ui, {}))
        proprio = self.humano_por_item.get(excluir_handle, {}).get(ui)
        if proprio:
            uso[proprio] -= 1                   # o proprio item nao conta
        tot = sum(max(0, n) for n in uso.values())
        votos = Counter()
        for hv, sim in (contexto or {}).get("vizinhos", ()):
            c = self.humano_por_item.get(hv, {}).get(ui)
            if c in caminhos:
                votos[c] += max(sim, 0.0) ** 4
        tv = sum(votos.values())
        cats = (contexto or {}).get("categorias", Counter())
        linhas = []
        for k, cam in enumerate(caminhos):
            n = max(0, uso.get(cam, 0))
            coer = cats.get(_categoria(cam), 0)
            if escolhido is not None and cam == escolhido:
                coer -= 1                       # no treino, a propria escolha nao conta
            linhas.append([n / tot if tot else 0.0, math.log1p(n), 1.0 if k == 0 else 0.0,
                           coer, votos.get(cam, 0.0) / tv if tv else 0.0, float(len(caminhos))])
        return np.array(linhas, dtype=np.float32)

    # -- escolha ---------------------------------------------------------------
    def escolher(self, ui, excluir_handle=None, caminho_humano=None, contexto=None):
        """(caminho, metodo)."""
        caminhos = self.vocab.caminhos(ui)
        if not caminhos:
            return None, "sem_arvore"
        if len(caminhos) == 1:
            return caminhos[0], "galho_unico"
        if caminho_humano and caminho_humano in caminhos:
            self.estatistica["humano"] += 1
            return caminho_humano, "escolha_do_catalogador"

        if self.modelo is not None and contexto is not None:
            X = self.caracteristicas(ui, caminhos, excluir_handle, contexto)
            p = self.modelo.predict_proba(X)[:, 1]
            self.estatistica["modelo"] += 1
            return caminhos[int(np.argmax(p))], "modelo_de_galho"

        uso = Counter({c: n for c, n in self.uso.get(ui, {}).items() if c in caminhos})
        if excluir_handle and ui in self.humano_por_item.get(excluir_handle, {}):
            uso[self.humano_por_item[excluir_handle][ui]] -= 1
        uso = +uso
        if uso:
            self.estatistica["acervo"] += 1
            melhor = max(uso.values())
            # empate: vence a ordem da BIREME
            return next(c for c in caminhos if uso.get(c) == melhor), "precedente_do_acervo"

        self.estatistica["bireme"] += 1
        return caminhos[0], "primeira_arvore_bireme"

    # -- treino e avaliacao ------------------------------------------------------
    def dados_de_treino(self, vetores_por_handle):
        """(X, y, grupos, item) de todas as escolhas humanas de galho."""
        Xs, ys, gs, its = [], [], [], []
        for ih, (h, mapa) in enumerate(self.humano_por_item.items()):
            if h not in vetores_por_handle:
                continue
            ctx = self.contexto(h, valores_humanos(self.acervo[h]), vetores_por_handle[h])
            for iu, (ui, escolhido) in enumerate(mapa.items()):
                caminhos = self.vocab.caminhos(ui)
                if escolhido not in caminhos:
                    continue
                Xs.append(self.caracteristicas(ui, caminhos, h, ctx, escolhido=escolhido))
                ys += [1 if c == escolhido else 0 for c in caminhos]
                gs += [(ih, iu)] * len(caminhos)
                its += [ih] * len(caminhos)
        return np.vstack(Xs), np.array(ys), gs, np.array(its)

    def treinar_e_avaliar(self, vetores_por_handle, dobras=5):
        """Validacao cruzada por item + modelo final com tudo. Devolve (metricas, n)."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        X, y, g, its = self.dados_de_treino(vetores_por_handle)
        acertos = Counter()
        total = 0
        for k in range(dobras):
            tr, te = its % dobras != k, its % dobras == k
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               random_state=0).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            por = defaultdict(list)
            for gg, pp, yy, xx in zip([g[i] for i in np.where(te)[0]], p, y[te], X[te]):
                por[gg].append((pp, yy, xx))
            for cands in por.values():
                total += 1
                acertos["combinado"] += max(cands, key=lambda c: c[0])[1]
                acertos["precedente"] += max(cands, key=lambda c: (c[2][0], c[2][2]))[1]
                acertos["primeira_bireme"] += max(cands, key=lambda c: c[2][2])[1]
        self.modelo = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                     random_state=0).fit(X, y)
        with open(ARQ_MODELO_GALHO, "wb") as f:
            pickle.dump({"modelo": self.modelo, "nomes": NOMES}, f)
        return {k: v / total for k, v in acertos.items()}, total
