# -*- coding: utf-8 -*-
"""Descritores raros: menos peso para a indexacao humana quando ha poucos exemplos, mais peso para o conhecimento
do proprio DeCS (pedido do responsavel, 2026-10-21; DECISOES §19 e publicacoes/artigo_decs).

Problema medido: 98% dos 10.822 descritores que os catalogadores usaram no Arca tem menos de 100 exemplos, e
juntos respondem por 55% das atribuicoes humanas. O classificador (gradient boosting) aprende com os mesmos sinais
para um descritor com 2.953 exemplos ("Brasil") e para um com 47 ("Eugenia"); nos raros a probabilidade satura
(ex.: Eugenia com 0,999 em textos sem relacao). Cortar os raros jogaria fora mais da metade do vocabulario.

Tres passos, cada um medido contra os catalogadores (regra 8) em ferramentas/decs/treinar_avaliar.py:
 1. MISTURA por frequencia: p = w(n) * p_humano + (1 - w(n)) * p_conhecimento, com w(n) = n / (n + K).
    p_humano e o classificador de sempre (kNN, rede, lexico, denso, prior). p_conhecimento e um classificador que
    so ve sinais do PROPRIO DeCS (nome e sinonimos no texto, similaridade do texto com nome + nota de escopo +
    sinonimos, numero de galhos): nao memoriza descritor nenhum, entao generaliza para os raros. n = exemplos
    humanos do descritor no treino. K sai da validacao (nunca do teste).
 2. CALIBRACAO por faixa de frequencia: regressao isotonica ajustada em previsoes fora da amostra (validacao
    cruzada no treino), separada por faixa de n. Assim 0,86 quer dizer "acerta 86%" tambem nos raros.
 3. LLM local (opcional, config.DECS_LLM_RAROS): antes de gravar sozinho um descritor raro (n < K_LLM), o LLM le
    o documento e a definicao do descritor e confirma; sem confirmacao, vira sugestao para revisao.
Nada disso aprende com saida de IA (regra 7): o gabarito continua sendo so o humano; a definicao e texto da BIREME.
"""
import numpy as np

from . import config

# faixas de frequencia (exemplos humanos do descritor no treino) usadas na calibracao e nos relatorios
FAIXAS = [(0, 4, "0-4"), (5, 29, "5-29"), (30, 99, "30-99"), (100, 299, "100-299"), (300, 10 ** 9, "300+")]
CONHECIMENTO = ["lex_kw_nome", "lex_kw_sin", "lex_tit_nome", "lex_tit_sin", "lex_res_nome", "lex_res_sin",
                "lex_res_freq", "denso", "denso_rank", "n_caminhos"] + (["lex_kw_aprox", "lex_kw_aprox_fraco"] if config.DECS_KW_APROXIMADO else [])
K_CANDIDATOS = [0, 3, 10, 30, 100]          # 0 = sem mistura (so o classificador humano)
K_LLM = 30                                  # abaixo disso, o LLM confirma antes de gravar sozinho (passo 3)


def faixa(n):
    for a, b, nome in FAIXAS:
        if a <= n <= b:
            return nome
    return FAIXAS[-1][2]


def peso_humano(n, K):
    return 1.0 if K <= 0 else n / (n + K)


class Ajuste:
    """O que o treino decide e a producao aplica: modelo de conhecimento, K e calibradores por faixa."""

    def __init__(self, colunas, modelo_conhecimento, K, calibradores):
        self.colunas = colunas                          # indices de CONHECIMENTO em CARACTERISTICAS
        self.modelo_conhecimento = modelo_conhecimento
        self.K = K
        self.calibradores = calibradores                # {faixa: IsotonicRegression}

    def misturar(self, p_humano, X, ns):
        if self.K <= 0 or self.modelo_conhecimento is None:
            return np.asarray(p_humano, dtype=float)
        pc = self.modelo_conhecimento.predict_proba(X[:, self.colunas])[:, 1]
        w = np.array([peso_humano(n, self.K) for n in ns])
        return w * np.asarray(p_humano) + (1 - w) * pc

    def calibrar(self, p, ns):
        p = np.asarray(p, dtype=float).copy()
        for i, n in enumerate(ns):
            c = self.calibradores.get(faixa(n))
            if c is not None:
                p[i] = float(c.predict([p[i]])[0])
        return p

    def aplicar(self, p_humano, X, ns):
        return self.calibrar(self.misturar(p_humano, X, ns), ns)


def ajustar_calibradores(p, y, ns, minimo=200):
    """Isotonica por faixa (so onde ha exemplos suficientes; faixa pequena junta-se a vizinha de cima)."""
    from sklearn.isotonic import IsotonicRegression
    f = np.array([faixa(n) for n in ns])
    cal = {}
    for _a, _b, nome in FAIXAS:
        sel = f == nome
        if sel.sum() < minimo or y[sel].sum() < 10:
            continue
        r = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        r.fit(p[sel], y[sel])
        cal[nome] = r
    return cal
