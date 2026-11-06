# -*- coding: utf-8 -*-
"""Rede multirrotulo: texto do documento -> probabilidade de cada descritor.

E a abordagem classica de indexacao MeSH/DeCS automatica ("extreme
multi-label classification"): uma camada oculta sobre o embedding do
documento, uma saida por descritor, treinada com a indexacao dos
catalogadores. Complementa o kNN: o kNN olha 30 vizinhos; a rede resume o
acervo inteiro.

SEM VAZAMENTO (validacao cruzada em 5 partes):
  os itens indexados sao divididos em 5 partes; a rede k e treinada sem a
  parte k. A nota de um item indexado vem SEMPRE da rede que nao o viu — no
  treino do classificador, na avaliacao e na producao. Itens sem indexacao
  (ou de teste) recebem a media das 5 redes, nenhuma das quais os viu.
"""
import os
import random

import numpy as np
import torch

from . import config

ARQ_REDE = os.path.join(config.DADOS_LOCAIS, "rede_decs.pt")
MIN_EXEMPLOS = 3          # descritor precisa de 3+ itens para ter saida propria
PARTES = 5
# Ajustado em 2026-10-02 (2.000 itens de validacao):
#   sem padronizar a entrada ........ P@1 0,084   (a rede nao aprende: os
#                                                 vetores e5 sao muito parecidos
#                                                 entre si)
#   padronizada, oculta 1024, 1e-3 .. P@1 0,598
#   padronizada, oculta 2048, 3e-3 .. P@1 0,635   <- usado
#   idem com 50 epocas .............. P@1 0,633
OCULTA = 2048
EPOCAS = 30
LOTE = 256
TAXA = 3e-3


class _Rede(torch.nn.Module):
    def __init__(self, dim, n_rotulos):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(dim, OCULTA), torch.nn.GELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(OCULTA, n_rotulos))

    def forward(self, x):
        return self.f(x)


def _treinar_uma(X, Y, dev, semente):
    torch.manual_seed(semente)
    rede = _Rede(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(rede.parameters(), lr=TAXA, weight_decay=1e-4)
    n = X.shape[0]
    passos = EPOCAS * ((n + LOTE - 1) // LOTE)
    agenda = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=TAXA, total_steps=passos)
    perda = torch.nn.BCEWithLogitsLoss()
    rede.train()
    for _ in range(EPOCAS):
        ordem = torch.randperm(n, device=dev)
        for i in range(0, n, LOTE):
            j = ordem[i:i + LOTE]
            opt.zero_grad()
            l = perda(rede(X[j]), Y[j])
            l.backward()
            opt.step()
            agenda.step()
    rede.eval()
    return rede.cpu().state_dict()


class RedeDecs:
    """As 5 redes + qual parte cada item indexado ficou de fora."""

    def __init__(self, rotulos, estados, parte_de, dim, media, desvio):
        self.rotulos = rotulos
        self.idx = {u: i for i, u in enumerate(rotulos)}
        self.parte_de = parte_de
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.media = media.float().to(self.dev)
        self.desvio = desvio.float().to(self.dev)
        self.redes = []
        for st in estados:
            r = _Rede(dim, len(rotulos))
            r.load_state_dict(st)
            self.redes.append(r.to(self.dev).eval())

    @torch.no_grad()
    def prever(self, vetor, handle=None):
        """Vetor de probabilidades (len(rotulos)) para um documento."""
        x = ((vetor.float().to(self.dev) - self.media) / self.desvio).unsqueeze(0)
        k = self.parte_de.get(handle)
        if k is not None:
            return torch.sigmoid(self.redes[k](x))[0].cpu().numpy()
        return np.mean([torch.sigmoid(r(x))[0].cpu().numpy() for r in self.redes], axis=0)

    def salvar(self, caminho=ARQ_REDE):
        torch.save({"rotulos": self.rotulos, "parte_de": self.parte_de,
                    # copia na CPU sem mover a rede (Module.cpu() move no lugar)
                    "estados": [{k: v.detach().cpu() for k, v in r.state_dict().items()}
                                for r in self.redes],
                    "dim": self.redes[0].f[1].in_features,
                    "media": self.media.cpu(), "desvio": self.desvio.cpu()}, caminho)


def treinar(vetores, handles, gabarito, excluir=(), progresso=print):
    """Treina as 5 redes com os itens indexados que nao estao em `excluir`."""
    excluir = set(excluir)
    itens = [h for h in handles if gabarito.get(h) and h not in excluir]
    from collections import Counter
    freq = Counter(u for h in itens for u in gabarito[h])
    rotulos = sorted(u for u, c in freq.items() if c >= MIN_EXEMPLOS)
    idx = {u: i for i, u in enumerate(rotulos)}
    pos = {h: i for i, h in enumerate(handles)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = vetores[[pos[h] for h in itens]].float().to(dev)
    # padronizar e o que faz a rede aprender (ver o cabecalho)
    media, desvio = X.mean(0), X.std(0) + 1e-6
    X = (X - media) / desvio
    Y = torch.zeros((len(itens), len(rotulos)), device=dev)
    for i, h in enumerate(itens):
        for u in gabarito[h]:
            if u in idx:
                Y[i, idx[u]] = 1.0
    ordem = list(range(len(itens)))
    random.Random(20261002).shuffle(ordem)
    parte_de = {itens[i]: n % PARTES for n, i in enumerate(ordem)}
    estados = []
    for k in range(PARTES):
        sel = torch.tensor([i for i, h in enumerate(itens) if parte_de[h] != k], device=dev)
        estados.append(_treinar_uma(X[sel], Y[sel], dev, semente=k))
        progresso("  rede %d/%d treinada (%d itens, %d descritores)"
                  % (k + 1, PARTES, len(sel), len(rotulos)))
    return RedeDecs(rotulos, estados, parte_de, X.shape[1], media, desvio)


def carregar(caminho=ARQ_REDE):
    if not os.path.exists(caminho):
        return None
    d = torch.load(caminho)
    return RedeDecs(d["rotulos"], d["estados"], d["parte_de"], d["dim"], d["media"], d["desvio"])
