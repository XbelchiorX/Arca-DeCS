# -*- coding: utf-8 -*-
"""Atribuicao de descritores DeCS a um documento.

Metodo (o mesmo desenho dos indexadores automaticos de producao, como o
MTI/MTIX da NLM e o DeCS Finder IA da BIREME): APRENDER COM A INDEXACAO
HUMANA, nao so comparar o texto com o nome do descritor.

Quatro fontes de candidatos, cada uma vira uma caracteristica:

1. kNN  — os documentos do proprio ARCA mais parecidos com este, ja
          indexados por catalogadores, "votam" com os descritores deles.
          E o sinal mais forte medido (ver docs/DECISOES.md).
2. LEXICO — o nome ou sinonimo do descritor aparece nas palavras-chave, no
          titulo ou no resumo. Nome preferido conta mais que sinonimo.
3. REDE  — rede multirrotulo treinada com o acervo indexado (decs_rede.py):
          texto -> probabilidade de cada descritor.
4. DENSO — similaridade entre o texto do documento e o texto do descritor
          (nome + nota de escopo + sinonimos). Era o unico sinal do pipeline
          antigo; aqui e so mais uma caracteristica.

Um classificador (gradient boosting) treinado nos itens do acervo combina as
caracteristicas e devolve a PROBABILIDADE de um catalogador ter atribuido
aquele descritor. O limiar de gravacao automatica e escolhido pela precisao
medida no conjunto de teste (config.DECS_PRECISAO_ALVO), nao no chute.
"""
import math
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
import torch

from . import config, decs_acervo, decs_embeddings
from .decs_interpretar import (CAMINHO_ANTIGO, CAMINHO_VALIDO, COMBINACAO, NOME,
                               SINONIMO, interpretar)
from .decs_vocabulario import norm

# Tipos de interpretacao confiaveis o bastante para servir de gabarito
TIPOS_GABARITO = {CAMINHO_VALIDO, CAMINHO_ANTIGO, NOME, SINONIMO, COMBINACAO}

KNN_K = 30
KNN_POTENCIA = 4          # sim**4 enfatiza os vizinhos realmente proximos
DENSO_TOP = 20
MAX_NGRAMA = 8

CARACTERISTICAS = [
    "knn_soma", "knn_frac", "knn_max", "knn_rank", "knn10_soma",
    "lex_kw_nome", "lex_kw_sin", "lex_tit_nome", "lex_tit_sin",
    "lex_res_nome", "lex_res_sin", "lex_res_freq",
    "denso", "denso_rank", "prior", "n_caminhos", "rede", "rede_rank",
] + (["lex_kw_aprox", "lex_kw_aprox_fraco"] if config.DECS_KW_APROXIMADO else [])
REDE_TOP = 30


# ---------------------------------------------------------------------------
# Gabarito: os descritores que o catalogador atribuiu a cada item
# ---------------------------------------------------------------------------
def valores_humanos(item):
    """Os valores de DECS que um CATALOGADOR escolheu para o item.

    Se o script ja gravou no item, o dc.subject.decs atual e saida da IA; o
    humano esta na linha [DECS-ORIGINAL] (coletada como `decs_original`).
    Aprender com a propria saida faria o modelo se reforcar sozinho.
    """
    orig = item.get("decs_original")
    return orig if orig is not None else item.get("decs", [])


def descritores_humanos(item, vocab):
    """UIs que os valores humanos do item representam com seguranca."""
    uis = []
    for v in valores_humanos(item):
        r = interpretar(v, vocab)
        if r["tipo"] in TIPOS_GABARITO and r["ui"] and r["ui"] not in uis:
            uis.append(r["ui"])
    return uis


def peso_qualidade(item, vocab):
    """Peso do item como exemplo de treino (opcao config.DECS_PESO_QUALIDADE, 05/10; a medir).

    15 anos de indexacao trazem exemplos de qualidade desigual. Dois sinais simples, sem olhar a resposta
    do modelo: (1) fracao dos valores humanos que se interpretam com seguranca como descritor do DeCS 2026
    (valor antigo/renomeado/solto pesa menos); (2) indexacao rasa (um descritor so) pesa metade.
    """
    vals = valores_humanos(item)
    if not vals:
        return 0.0
    ok = len(descritores_humanos(item, vocab))
    peso = max(ok / len(vals), 0.2)
    if ok <= 1:
        peso *= 0.5
    return round(peso, 3)


# ---------------------------------------------------------------------------
# Lexico: casamento de nome/sinonimo no texto por n-gramas
# ---------------------------------------------------------------------------
def _tokens(texto):
    return re.findall(r"\w+(?:[-']\w+)*", norm(texto))


def casamentos_lexicos(texto, vocab):
    """{ui: (n_nome, n_sinonimo)} para todo n-grama do texto que e termo DeCS.

    Sinonimo ambiguo (aponta para varios descritores) conta para todos,
    mas so como sinonimo: quem decide e o classificador, com o contexto.
    """
    toks = _tokens(texto)
    achados = defaultdict(lambda: [0, 0])
    for i in range(len(toks)):
        for n in range(1, MAX_NGRAMA + 1):
            if i + n > len(toks):
                break
            g = " ".join(toks[i:i + n])
            if g in vocab.nomes:
                for ui in vocab.nomes[g]:
                    achados[ui][0] += 1
            elif g in vocab.sinonimos:
                for ui in vocab.sinonimos[g]:
                    achados[ui][1] += 1
    return achados


def _kw_textos(item):
    return [(x.get("v") or "").strip() for x in item.get("keywords", []) if (x.get("v") or "").strip()]


def casamentos_palavras_chave(item, vocab):
    """Palavra-chave inteira que E um termo DeCS (nao n-grama: o campo todo).

    Com DECS_KW_APROXIMADO: {ui: [nome, sinonimo, aproximado, aproximado fraco]}. O nivel 'variante' do
    vocabulario (ordem sem conjuncao: "Education and Health" = "Health Education") da lugar ao casamento
    aproximado com travas (arca/decs_palavra_chave.py)."""
    from .decs_palavra_chave import FRACOS
    achados = defaultdict(lambda: [0, 0, 0, 0])
    for kw in _kw_textos(item):
        uis, nivel = vocab.uis_do_termo(kw)
        if config.DECS_KW_APROXIMADO and nivel not in ("nome", "sinonimo"):
            uis, nivel = _indice_aproximado(vocab).uis(kw)
        for ui in uis:
            achados[ui][{"nome": 0, "sinonimo": 1, "variante": 1}.get(nivel, 3 if nivel in FRACOS else 2)] += 1
    return achados


def _indice_aproximado(vocab):
    if getattr(vocab, "_aproximado", None) is None:
        from .decs_palavra_chave import IndiceAproximado
        vocab._aproximado = IndiceAproximado(vocab)
    return vocab._aproximado


# ---------------------------------------------------------------------------
# Indice de documentos (kNN) e de descritores (denso)
# ---------------------------------------------------------------------------
class Indexador:
    """Guarda tudo que e preciso para pontuar candidatos de um item."""

    def __init__(self, vocab, acervo, vetores_itens, handles, vetores_descr, uis_descr,
                 gabarito, modelo=None, limiar=None, rede=None, ajuste=None):
        self.vocab = vocab
        self.acervo = acervo
        self.handles = handles
        self.pos = {h: i for i, h in enumerate(handles)}
        self.V = vetores_itens                                 # (n, d) float16
        self.D = vetores_descr                                 # (m, d) float16
        self.uis_descr = uis_descr
        self.gabarito = gabarito                               # handle -> [ui]
        self.modelo = modelo
        self.limiar = limiar
        self.rede = rede
        self.ajuste = ajuste                                   # decs_raros.Ajuste (mistura + calibracao) ou None
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # so itens com gabarito servem de vizinho
        self.idx_viz = torch.tensor([self.pos[h] for h in handles if gabarito.get(h)])
        self.Vviz = self.V[self.idx_viz].to(self.dev)
        self.Dg = self.D.to(self.dev)
        self.recalcular_prior()

    def recalcular_prior(self, excluir=()):
        """Frequencia de cada descritor no acervo indexado (log).

        Na avaliacao, os itens de teste ficam de fora: senao, para um
        descritor raro, o proprio item de teste aumentaria a frequencia do
        descritor que ele tem — vazamento da resposta.
        """
        excluir = set(excluir)
        hs = [h for h in self.handles if self.gabarito.get(h) and h not in excluir]
        self.freq = Counter(u for h in hs for u in self.gabarito[h])
        self.excluidos_prior = excluir
        self.tot = max(1, len(hs))

    def _prior(self, u, proprios):
        """log da frequencia, sem contar os itens excluidos da propria consulta."""
        c = self.freq.get(u, 0) - sum(1 for h in proprios
                                      if h not in self.excluidos_prior
                                      and u in self.gabarito.get(h, ()))
        return math.log((max(c, 0) + 1) / self.tot)

    # -- vizinhos -------------------------------------------------------------
    def vizinhos(self, vetor, excluir=(), k=KNN_K):
        """[(handle, similaridade)] dos k itens indexados mais parecidos."""
        s = (self.Vviz @ vetor.to(self.dev).unsqueeze(1)).squeeze(1).float()
        top = torch.topk(s, k=min(k + len(excluir) + 5, s.numel()))
        saida = []
        for sc, j in zip(top.values.tolist(), top.indices.tolist()):
            h = self.handles[int(self.idx_viz[j])]
            if h in excluir:
                continue
            saida.append((h, sc))
            if len(saida) == k:
                break
        return saida

    # -- candidatos + caracteristicas ----------------------------------------
    def caracteristicas(self, item, vetor, excluir=(), extras=(), proprio=None):
        """{ui: vetor de caracteristicas} para todos os candidatos do item.

        `extras`: descritores que devem ser pontuados mesmo que nenhuma fonte
        os tenha proposto (ex.: as opcoes de um termo ambiguo do catalogador).
        """
        cand = defaultdict(dict)
        for u in extras:
            cand[u]

        viz = self.vizinhos(vetor, excluir)
        soma = defaultdict(float)
        soma10 = defaultdict(float)     # so os 10 mais proximos: sinal mais "local"
        cont = Counter()
        smax = defaultdict(float)
        for pos_viz, (h, s) in enumerate(viz):
            w = max(s, 0.0) ** KNN_POTENCIA
            for u in self.gabarito.get(h, []):
                soma[u] += w
                if pos_viz < 10:
                    soma10[u] += w
                cont[u] += 1
                smax[u] = max(smax[u], s)
        total_w = sum(max(s, 0.0) ** KNN_POTENCIA for _h, s in viz) or 1.0
        total_w10 = sum(max(s, 0.0) ** KNN_POTENCIA for _h, s in viz[:10]) or 1.0
        ordem = sorted(soma, key=soma.get, reverse=True)
        for r, u in enumerate(ordem[:60]):
            cand[u].update(knn_soma=soma[u] / total_w, knn_frac=cont[u] / max(1, len(viz)),
                           knn_max=smax[u], knn_rank=r + 1,
                           knn10_soma=soma10.get(u, 0.0) / total_w10)

        kw = casamentos_palavras_chave(item, self.vocab)
        tit = casamentos_lexicos(" ".join(x["v"] for x in item.get("titulos", [])), self.vocab)
        res = casamentos_lexicos(" ".join(x["v"] for x in item.get("resumos", [])), self.vocab)
        for u, (a, b, c, d) in kw.items():
            cand[u].update(lex_kw_nome=a, lex_kw_sin=b, lex_kw_aprox=c, lex_kw_aprox_fraco=d)
        for u, (a, b) in tit.items():
            cand[u].update(lex_tit_nome=a, lex_tit_sin=b)
        for u, (a, b) in res.items():
            cand[u].update(lex_res_nome=a, lex_res_sin=b, lex_res_freq=a + b)

        prob_rede = None
        if self.rede is not None:
            # `proprio`: o handle do item, para usar a rede que NAO o viu
            prob_rede = self.rede.prever(vetor, proprio)
            for r, j in enumerate(np.argsort(-prob_rede)[:REDE_TOP]):
                cand[self.rede.rotulos[j]].update(rede_rank=r + 1)

        sd = (self.Dg @ vetor.to(self.dev).unsqueeze(1)).squeeze(1).float()
        top = torch.topk(sd, k=DENSO_TOP)
        for r, (sc, j) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            cand[self.uis_descr[j]].update(denso_rank=r + 1)
        # similaridade densa de TODOS os candidatos (nao so do top-20)
        idx = {u: i for i, u in enumerate(self.uis_descr)}
        saida = {}
        for u, f in cand.items():
            if u not in idx or not self.vocab.caminhos(u):
                continue
            f.setdefault("denso", float(sd[idx[u]]))
            f.setdefault("knn_rank", 99)
            f.setdefault("denso_rank", 99)
            f.setdefault("rede_rank", 99)
            if prob_rede is not None and u in self.rede.idx:
                f["rede"] = float(prob_rede[self.rede.idx[u]])
            f["prior"] = self._prior(u, excluir)
            f["n_caminhos"] = len(self.vocab.caminhos(u))
            saida[u] = [float(f.get(c, 0.0)) for c in CARACTERISTICAS]
        return saida

    # -- pontuacao ------------------------------------------------------------
    def pontuar(self, item, vetor, excluir=(), extras=(), proprio=None):
        """[(ui, probabilidade)] ordenado, para um item.

        O que o catalogador ja atribuiu NAO e caracteristica do modelo (seria
        entregar a resposta); o tratamento dos descritores humanos e feito
        fora daqui, em decs_correcao.
        """
        feats = self.caracteristicas(item, vetor, excluir, extras, proprio)
        if not feats:
            return []
        uis = list(feats)
        X = np.array([feats[u] for u in uis], dtype=np.float32)
        p = self.modelo.predict_proba(X)[:, 1]
        if self.ajuste is not None:
            # descritores raros (decs_raros, DECISOES §19): n = exemplos humanos, sem contar os itens excluidos
            ns = [self.freq.get(u, 0) - sum(1 for h in excluir if h not in self.excluidos_prior
                                            and u in self.gabarito.get(h, ())) for u in uis]
            p = self.ajuste.aplicar(p, X, ns)
        return sorted(zip(uis, [float(x) for x in p]), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Construcao (embeddings + gabarito) e persistencia do modelo
# ---------------------------------------------------------------------------
ARQ_EMB_ITENS = os.path.join(config.CACHE, "emb_itens_%s.pt" % config.MODELO_EMBEDDING)
ARQ_EMB_DESCR = os.path.join(config.CACHE, "emb_descritores_%s.pt" % config.MODELO_EMBEDDING)
ARQ_MODELO = os.path.join(config.DADOS_LOCAIS, "modelo_decs.pkl")


def embeddings_descritores(vocab):
    uis = sorted(u for u in vocab.base if vocab.caminhos(u))
    textos = [vocab.texto_descritor(u) for u in uis]
    vet = decs_embeddings.embutir_com_cache(uis, textos, ARQ_EMB_DESCR, mostrar_progresso=False)
    return uis, vet


def embeddings_itens(acervo):
    handles = sorted(acervo)
    textos = [decs_acervo.texto_do_item(acervo[h], decs_embeddings.MAX_RESUMO) for h in handles]
    vet = decs_embeddings.embutir_com_cache(handles, textos, ARQ_EMB_ITENS)
    return handles, vet


def montar_indexador(vocab, acervo, com_modelo=True):
    handles, V = embeddings_itens(acervo)
    uis, D = embeddings_descritores(vocab)
    # o 8B roda num llama-server que ocupa quase toda a GPU: solta antes de o indice ir para a GPU
    decs_embeddings.liberar()
    gabarito = {h: descritores_humanos(acervo[h], vocab) for h in handles}
    modelo = limiar = rede = ajuste = None
    if com_modelo:
        from . import decs_rede
        rede = decs_rede.carregar()
    if com_modelo and os.path.exists(ARQ_MODELO):
        with open(ARQ_MODELO, "rb") as f:
            salvo = pickle.load(f)
        if salvo.get("caracteristicas") != CARACTERISTICAS:
            raise RuntimeError("Modelo salvo e de outra versao das caracteristicas. "
                               "Rode: python ferramentas/decs/treinar_avaliar.py")
        modelo, limiar, ajuste = salvo["modelo"], salvo["limiar"], salvo.get("ajuste")
    return Indexador(vocab, acervo, V, handles, D, uis, gabarito, modelo, limiar, rede, ajuste)


def salvar_modelo(modelo, limiar, metricas, ajuste=None):
    with open(ARQ_MODELO, "wb") as f:
        pickle.dump({"modelo": modelo, "limiar": limiar, "metricas": metricas, "ajuste": ajuste,
                     "caracteristicas": CARACTERISTICAS}, f)
