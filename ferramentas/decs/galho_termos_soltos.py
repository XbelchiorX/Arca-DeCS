# -*- coding: utf-8 -*-
"""Galho para os termos SOLTOS de dc.subject.decs (pedido do responsavel, 2026-10-17).

Referencia do galho: o que os catalogadores ja gravaram (decisao do responsavel). Modelo: o modelo de galho de
sempre (costume do acervo + coerencia com os outros descritores do item + documentos parecidos), treinado com as
6.090 escolhas humanas de galho (medido: 82,3% de acerto; >= 0,70 de confianca cobre 70% com 91,5% —
saidas/avaliacao/galho_conteudo_e5-large.md). Salvo a parte em DADOS_LOCAIS/modelo_galho_soltos.pkl (o modelo em
uso pela D1-D3 nao muda).
  - galho unico: grava o caminho (nao ha escolha);
  - varios galhos e confianca >= --limiar: grava o galho do modelo;
  - abaixo do limiar: NAO mexe; o relatorio mostra a escolha do modelo e o galho mais usado pelos catalogadores.
Termo com qualificador ("Tuberculose/epidemiologia"): o qualificador e descartado (decisao de 2026-10-02).
Se o descritor ja esta no item com galho, o termo solto so sai (nao duplica). [DECS-ORIGINAL] antes da 1a gravacao.
  --completar (opcao (a), decisao do responsavel 2026-10-17): abaixo do limiar grava o galho mais usado pelos
  catalogadores para o descritor, marcado como "mais_usado" no relatorio (nenhum termo DeCS fica sem galho).
Uso: python ferramentas/decs/galho_termos_soltos.py --item <handle|uuid|url> [...] | --lista arq [--limiar 0.7] [--completar] [--gravar]
"""
import argparse
import csv
import os
import pickle
import sys
from collections import Counter
from datetime import datetime

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
# hibrido (decisao do responsavel, 2026-10-19): o conteudo do documento e o texto do galho no mesmo espaco do 8B
os.environ.setdefault("DECS_MODELO_EMBEDDING", "qwen3-8b")
from arca import config  # noqa: E402
from arca import decs_rastreabilidade as R  # noqa: E402

ARQ_MODELO = os.path.join(config.DADOS_LOCAIS, "modelo_galho_soltos%s.pkl" % (
    "" if config.MODELO_EMBEDDING == "e5-large" else "_" + config.MODELO_EMBEDDING))
ARQ_GALHOS = os.path.join(config.CACHE, "texto_galhos_%s.pt" % config.MODELO_EMBEDDING)
TIPOS_TERMO = ("nome", "sinonimo", "combinacao_oficial")
MEXE = ("galho_unico", "conteudo", "modelo", "mais_usado", "retirar_duplicado")


class Galhos:
    def __init__(self):
        import torch
        from arca import decs_acervo, decs_galho, decs_indexador as IX, decs_vocabulario as V
        self.vocab = V.carregar_vocabulario()
        acervo = decs_acervo.carregar_acervo()
        c = torch.load(IX.ARQ_EMB_ITENS, weights_only=False)
        pos = {k: i for i, k in enumerate(c["chaves"])}
        hs = [h for h in acervo if h in pos]
        self.vet = {h: v for h, v in zip(hs, torch.nn.functional.normalize(
            c["vetores"][[pos[h] for h in hs]].float(), dim=1))}
        V_ = torch.stack([self.vet[h] for h in hs])
        self.g = decs_galho.EscolhaDeGalho(self.vocab, acervo, vetores=V_, handles=hs, carregar_modelo=False)
        if os.path.exists(ARQ_MODELO):
            self.modelo = pickle.load(open(ARQ_MODELO, "rb"))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            X, y, _g, _i = self.g.dados_de_treino(self.vet)
            self.modelo = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X, y)
            pickle.dump(self.modelo, open(ARQ_MODELO, "wb"))
        import threading
        self.vcam = torch.load(ARQ_GALHOS, weights_only=False) if os.path.exists(ARQ_GALHOS) else {}
        self._trava = threading.Lock()

    def preparar(self, uis=(), itens=()):
        """Vetores que faltam, de uma vez: texto dos galhos dos descritores `uis` e documentos [(h, item)] fora do cache."""
        import torch
        from arca import decs_acervo, decs_embeddings
        with self._trava:
            cams = sorted({c for u in uis for c in self.vocab.caminhos(u)} - set(self.vcam))
            docs = [(h, it) for h, it in itens if h not in self.vet]
            # em blocos, salvando a cada um: se o servidor cair, o que ja foi feito nao se perde (2026-10-19)
            for k in range(0, len(cams), 500):
                bloco = cams[k:k + 500]
                v = torch.nn.functional.normalize(decs_embeddings.embutir(
                    [c.replace("::", " > ") for c in bloco]).float(), dim=1)
                self.vcam.update({c: v[i] for i, c in enumerate(bloco)})
                torch.save(self.vcam, ARQ_GALHOS)
                if len(cams) > 500:
                    print("  textos dos galhos: %d/%d" % (min(k + 500, len(cams)), len(cams)), flush=True)
            if docs:
                v = torch.nn.functional.normalize(decs_embeddings.embutir(
                    [decs_acervo.texto_do_item(it, decs_embeddings.MAX_RESUMO) for _h, it in docs],
                    mostrar_progresso=len(docs) > 200).float(), dim=1)
                self.vet.update({h: v[i] for i, (h, _it) in enumerate(docs)})

    def conteudo(self, vdoc, ui):
        """Galho cujo texto mais se parece com o documento, a margem para o 2o e o 2o."""
        cams = self.vocab.caminhos(ui)
        if any(c not in self.vcam for c in cams):
            self.preparar([ui])
        s = sorted(((float(self.vcam[c] @ vdoc), c) for c in cams), reverse=True)
        return s[0][1], s[0][0] - s[1][0], s[1][1]

    def vetor(self, h, item):
        if h not in self.vet:
            self.preparar(itens=[(h, item)])
        return self.vet[h]

    def precedente(self, ui):
        uso = self.g.uso.get(ui, Counter())
        cams = self.vocab.caminhos(ui)
        return max(cams, key=lambda c: (uso.get(c, 0), -cams.index(c)))

    def escolher(self, h, ui, ctx):
        cams = self.vocab.caminhos(ui)
        X = self.g.caracteristicas(ui, cams, h, ctx)
        p = self.modelo.predict_proba(X)[:, 1]
        k = int(np.argmax(p))
        return cams[k], float(p[k] / (p.sum() or 1.0))


def planejar(G, h, item, vals, limiar, completar=False, margem=None):
    """margem (hibrido): se o conteudo separa o melhor galho do 2o por >= margem, o conteudo decide (acao
    "conteudo"); senao, modelo (>= limiar) ou costume da casa. Um galho por descritor; o 2o galho do conteudo vai
    para o relatorio (revisao humana, decisao de 2026-10-19), nunca gravado."""
    from arca.decs_interpretar import interpretar
    vdoc = G.vetor(h, item)
    ctx = G.g.contexto(h, vals, vdoc)
    ja = set(v for v in vals if "::" in v)
    uis_com_galho = {interpretar(v, G.vocab).get("ui") for v in ja} - {None}
    linhas, novos = [], set()
    for i, v in enumerate(vals):
        if "::" in v:
            continue
        r = interpretar(v, G.vocab)
        ui = r.get("ui")
        if r["tipo"] not in TIPOS_TERMO or not ui or not G.vocab.caminhos(ui):
            continue
        cams = G.vocab.caminhos(ui)
        if ui in uis_com_galho or ui in novos:
            linhas.append(dict(i=i, antes=v, depois="", acao="retirar_duplicado", conf="", modelo="", precedente=""))
            continue
        if len(cams) == 1:
            novos.add(ui)
            linhas.append(dict(i=i, antes=v, depois=cams[0], acao="galho_unico", conf="", modelo="", precedente=""))
            continue
        esc, conf = G.escolher(h, ui, ctx)
        prec = G.precedente(ui)
        base = dict(i=i, antes=v, conf="%.2f" % conf, modelo=esc, precedente=prec)
        if margem is not None:
            cont, marg, seg = G.conteudo(vdoc, ui)
            base.update(conteudo=cont, margem="%.4f" % marg, segundo=seg)
            if marg >= margem:
                novos.add(ui)
                linhas.append(dict(base, depois=cont, acao="conteudo"))
                continue
        if conf >= limiar:
            novos.add(ui)
            linhas.append(dict(base, depois=esc, acao="modelo"))
        elif completar:   # opcao (a) do responsavel: abaixo do limiar, o galho mais usado pelos catalogadores
            novos.add(ui)
            linhas.append(dict(base, depois=prec, acao="mais_usado"))
        else:
            linhas.append(dict(base, depois="", acao="abaixo_do_limiar"))
    for l in linhas:
        for k in ("conf", "modelo", "precedente", "conteudo", "margem", "segundo"):
            l.setdefault(k, "")
    return linhas


def aplicar(cli, G, ident, limiar, gravar, completar=False, margem=None):
    from arca.decs_correcao import item_do_dspace
    obj = cli.item_por_handle(ident)
    if not obj:
        return None, "erro_item", []
    h = obj.get("handle")
    md = obj.get("metadata") or {}
    vals = [m.get("value", "") for m in md.get(config.CAMPO_DECS) or []]
    linhas = planejar(G, h, item_do_dspace(obj), vals, limiar, completar, margem)
    mexe = [l for l in linhas if l["acao"] in MEXE]
    if not mexe or not gravar:
        return h, ("sem_alteracao" if not mexe else "a_corrigir"), linhas
    ops = []
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(h)
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else vals)))
    esperado = list(vals)
    for l in mexe:
        if l["acao"] != "retirar_duplicado":
            ops.append({"op": "replace", "path": "/metadata/%s/%d/value" % (config.CAMPO_DECS, l["i"]),
                        "value": l["depois"]})
            esperado[l["i"]] = l["depois"]
    for l in sorted((l for l in mexe if l["acao"] == "retirar_duplicado"), key=lambda l: -l["i"]):
        ops.append({"op": "remove", "path": "/metadata/%s/%d" % (config.CAMPO_DECS, l["i"])})
        del esperado[l["i"]]
    cli.patch("/core/items/" + obj["uuid"], ops)
    depois = [x.get("value", "") for x in (cli.get("/core/items/" + obj["uuid"]).get("metadata") or {})
              .get(config.CAMPO_DECS) or []]
    R.registrar(h, vals, depois, [{"acao": "GALHO_" + l["acao"].upper(), "antes": l["antes"], "depois": l["depois"],
                                   "probabilidade": l["conf"]} for l in mexe], aplicado=True)
    return h, ("corrigido" if depois == esperado else "DIVERGENTE"), linhas


COLUNAS = ["handle", "antes", "depois", "acao", "confianca", "escolha_do_modelo", "mais_usado_catalogadores",
           "escolha_do_conteudo", "margem_conteudo", "segundo_galho_conteudo"]


def _linha(h, l):
    return [h, l["antes"], l["depois"], l["acao"], l["conf"], l["modelo"], l["precedente"], l["conteudo"],
            l["margem"], l["segundo"]]


def todos(G, a, w):
    """Acervo inteiro: mapeia o ARCA ao vivo, prepara de uma vez os vetores que faltam e aplica com 8 conexoes
    (cada item e lido de novo antes de gravar)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from arca import idiomas as I
    from arca.decs_correcao import item_do_dspace
    from arca.decs_interpretar import interpretar
    objs = I.mapear_objetos(I.criar_cliente(), 100, 8)
    alvo, uis, docs = [], set(), []
    for o in objs:
        vals = [m.get("value", "") for m in (o.get("metadata") or {}).get(config.CAMPO_DECS) or []]
        soltos = [interpretar(v, G.vocab) for v in vals if "::" not in v]
        us = [r["ui"] for r in soltos if r["tipo"] in TIPOS_TERMO and r.get("ui") and G.vocab.caminhos(r["ui"])]
        if not o.get("handle") or not us:
            continue
        alvo.append(o)
        multi = [u for u in us if len(G.vocab.caminhos(u)) > 1]
        uis.update(multi)
        if multi and o["handle"] not in G.vet:
            docs.append((o["handle"], item_do_dspace(o)))
    print("%d itens mapeados; %d com termo solto do DeCS; %d descritores com varios galhos; %d documentos sem vetor"
          % (len(objs), len(alvo), len(uis), len(docs)), flush=True)
    G.preparar(uis, docs)
    cont, acoes, trava = Counter(), Counter(), threading.Lock()

    def um(o):
        try:
            if a.gravar:
                return aplicar(I.get_thread_client()._c, G, o["handle"], a.limiar, True, a.completar, a.margem)
            vals = [m.get("value", "") for m in (o.get("metadata") or {}).get(config.CAMPO_DECS) or []]
            linhas = planejar(G, o["handle"], item_do_dspace(o), vals, a.limiar, a.completar, a.margem)
            return o["handle"], ("a_corrigir" if any(l["acao"] in MEXE for l in linhas) else "sem_alteracao"), linhas
        except Exception as e:
            print("  ERRO %s: %s" % (o.get("handle"), str(e)[:200]), flush=True)
            return o.get("handle"), "erro", []

    with ThreadPoolExecutor(8) as ex:
        for k, fut in enumerate(as_completed([ex.submit(um, o) for o in alvo]), 1):
            h, st, linhas = fut.result()
            with trava:
                cont[st] += 1
                for l in linhas:
                    w.writerow(_linha(h, l))
                    acoes[l["acao"]] += 1
                if k % 2000 == 0:
                    print("  %d/%d %s" % (k, len(alvo), dict(cont)), flush=True)
    print("FIM itens: %s | valores: %s" % (dict(cont), dict(acoes)))


def main():
    global MEXE
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", nargs="+")
    ap.add_argument("--todos", action="store_true", help="acervo inteiro (mapeamento ao vivo)")
    ap.add_argument("--limiar", type=float, default=0.70)
    ap.add_argument("--margem", type=float, default=0.010,
                    help="hibrido: margem do conteudo a partir da qual ele decide (medida no teste cego, 2026-10-19); "
                         "negativo desliga o conteudo")
    ap.add_argument("--gravar", action="store_true")
    ap.add_argument("--completar", action="store_true",
                    help="abaixo do limiar, grava o galho mais usado pelos catalogadores (nenhum termo DeCS sem galho)")
    ap.add_argument("--lista", help="arquivo com um handle/uuid/url por linha")
    ap.add_argument("--acoes", nargs="+", choices=MEXE,
                    help="grava so estas acoes (ex.: --acoes galho_unico); as outras ficam so na simulacao")
    a = ap.parse_args()
    if a.acoes:
        MEXE = tuple(a.acoes)
    if a.margem < 0:
        a.margem = None
    if a.lista:
        a.item = (a.item or []) + [l.strip() for l in open(a.lista, encoding="utf-8") if l.strip()]
    from arca import preparo
    if preparo.garantir():
        sys.exit("Dados locais faltando que nao puderam ser refeitos (veja acima).")
    G = Galhos()
    pasta = os.path.join(config.SAIDAS_DECS, "galhos")
    os.makedirs(pasta, exist_ok=True)
    saida = os.path.join(pasta, "%s%s_%s.csv" % ("gravacao" if a.gravar else "simulacao", "_acervo" if a.todos else "",
                                                datetime.now().strftime("%Y%m%d_%H%M%S")))
    with open(saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(COLUNAS)
        if a.todos:
            todos(G, a, w)
        else:
            from arca import dspace
            cli = dspace.conectar()
            for ident in a.item:
                ident = ident.rstrip("/").split("/")[-1] if "/items/" in ident else ident
                h, st, linhas = aplicar(cli, G, ident, a.limiar, a.gravar, a.completar, a.margem)
                print("\n%s: %s" % (h, st))
                for l in linhas:
                    w.writerow(_linha(h, l))
                    print("  [%s%s] %s\n      -> %s" % (l["acao"], " %s" % l["conf"] if l["conf"] else "", l["antes"],
                                                        l["depois"] or ("(fica solto; modelo: %s | mais usado: %s)"
                                                                        % (l["modelo"], l["precedente"])
                                                                        if l["acao"] == "abaixo_do_limiar" else "(sai: ja esta no item)")))
    print("\nrelatorio:", saida)


if __name__ == "__main__":
    try:
        main()
    finally:   # o llama-server do 8B nao pode ficar orfao segurando a GPU (aconteceu em 2026-10-19)
        from arca import decs_embeddings
        decs_embeddings.liberar()
