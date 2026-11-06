# -*- coding: utf-8 -*-
"""Corrige o DECS que o item tem e atribui o que falta. Um item por vez.

O PLANO de um item e uma lista de linhas, uma por decisao:

  MANTER            o valor ja e um caminho valido da arvore do servidor
  EXPANDIR          termo do catalogador -> caminho completo (mesmo descritor)
  ATUALIZAR         caminho/termo antigo, sinonimo ou combinacao oficial ->
                    caminho atual do descritor certo
  DESAMBIGUAR       termo que serve a varios descritores; o documento decidiu
  REVISAR           termo ambiguo sem decisao segura: o valor fica como esta
                    e vai para a lista de revisao
  FORA_DA_ARVORE    descritor DeCS sem lugar na arvore (Masculino, Feminino):
                    fica como esta
  PARA_PALAVRA_CHAVE termo que nao e DeCS: sai do campo de DECS e vai para as
                    palavras-chave (config.DECS_TERMO_FORA_DO_DECS)
  RETIRAR           qualificador solto ou duplicata
  ADICIONAR         descritor novo, probabilidade >= limiar automatico
  SUGERIR           descritor novo para revisao humana (NAO e gravado)

Principios:
  - O trabalho do catalogador e preservado: nada humano sai sem destino
    (vira caminho, vai para palavra-chave, ou fica) e o estado anterior e
    guardado em [DECS-ORIGINAL] antes da primeira gravacao.
  - So se grava valor que existe na arvore publicada no servidor.
  - Um descritor aparece uma vez so no item.
"""
import csv
import os
from collections import Counter
from datetime import datetime

from . import config
from . import decs_interpretar as I
from . import decs_rastreabilidade as R
from .decs_vocabulario import norm

GRAVA = {"MANTER", "EXPANDIR", "ATUALIZAR", "DESAMBIGUAR", "REVISAR", "FORA_DA_ARVORE", "ADICIONAR"}
MARGEM_DESAMBIGUAR = 0.15

CAMPOS_PLANO = ["handle", "titulo", "acao", "antes", "depois", "ui", "descritor",
                "metodo", "galho", "probabilidade", "motivo"]


class Contexto:
    """Tudo o que o planejamento precisa, carregado uma vez por execucao."""

    def __init__(self, vocab, indexador, galho, sugerir=True):
        self.vocab = vocab
        self.ix = indexador
        self.galho = galho
        self.sugerir = sugerir and indexador is not None and indexador.modelo is not None
        lim = (indexador.limiar if indexador and indexador.limiar else {}) or {}
        self.lim_auto = lim.get("automatico", 0.99)
        self.lim_revisao = lim.get("revisao", 0.99)


def _titulo(item):
    t = (item.get("titulos") or [{}])[0].get("v", "")
    return t[:120]


def _vetor(ctx, handle, item):
    """Vetor do documento: do cache se o texto nao mudou, senao calcula."""
    from . import decs_acervo, decs_embeddings
    texto = decs_acervo.texto_do_item(item, decs_embeddings.MAX_RESUMO)
    ix = ctx.ix
    if ix is not None and handle in ix.pos:
        return ix.V[ix.pos[handle]]
    return decs_embeddings.embutir([texto])[0]


def base_humana(handle, item, vocab=None):
    """Os valores sobre os quais o plano e feito.

    Item nunca processado: o dc.subject.decs atual (e humano por definicao).
    Item ja processado: o [DECS-ORIGINAL] (trabalho do catalogador) MAIS o que
    alguem acrescentou a mao depois da ultima gravacao do script — sem isso,
    reprocessar apagaria uma correcao humana feita no meio tempo. Se nao ha
    registro local da ultima gravacao, usa o atual inteiro (conservador: nao
    perde nada).

    "Acrescentado" e comparado por DESCRITOR quando ha vocabulario: o mesmo
    descritor escrito em outro formato (termo x caminho) nao e edicao humana.
    """
    atual = list(item.get("decs", []))
    original = item.get("decs_original")
    if original is None:
        return atual
    ultima = R.ultima_saida(handle)
    if ultima is None:
        return atual

    def chave(v):
        if vocab is not None:
            ui = I.interpretar(v, vocab)["ui"]
            if ui:
                return ui
        return norm(v)

    conhecidos = {chave(v) for v in list(ultima) + list(original)}
    acrescidos = [v for v in atual if chave(v) not in conhecidos]
    return list(original) + acrescidos


def planejar(handle, item, ctx):
    """Devolve (linhas, final, novas_palavras_chave)."""
    vocab = ctx.vocab
    titulo = _titulo(item)
    linhas = []
    final = []                 # valores na ordem
    uis_final = set()
    novas_kw = []

    vetor = None
    if ctx.ix is not None:
        vetor = _vetor(ctx, handle, item)

    def linha(acao, antes, depois="", ui="", metodo="", galho="", prob="", motivo=""):
        linhas.append({"handle": handle, "titulo": titulo, "acao": acao, "antes": antes,
                       "depois": depois, "ui": ui or "", "descritor": vocab.nome(ui) if ui else "",
                       "metodo": metodo, "galho": galho,
                       "probabilidade": ("%.3f" % prob) if isinstance(prob, float) else prob,
                       "motivo": motivo})

    def colocar(acao, antes, ui, metodo, caminho_humano=None, prob="", motivo=""):
        caminho, como = ctx.galho.escolher(ui, excluir_handle=handle,
                                           caminho_humano=caminho_humano,
                                           contexto=ctx_galho)
        if not caminho:
            linha("FORA_DA_ARVORE", antes, antes, ui, metodo, "", prob,
                  "descritor sem caminho na arvore do servidor")
            final.append(antes)
            return
        if ui in uis_final:
            linha("RETIRAR", antes, "", ui, metodo, como, prob, "descritor repetido no item")
            return
        uis_final.add(ui)
        final.append(caminho)
        if acao == "MANTER" and caminho != antes:
            acao = "ATUALIZAR"
        linha(acao, antes, caminho, ui, metodo, como, prob, motivo)

    # ---- 1. o que o item ja tem -------------------------------------------
    atuais = base_humana(handle, item, vocab)
    # contexto para o modelo de galho: categorias dos descritores do item e
    # documentos parecidos (calculado uma vez por item)
    ctx_galho = ctx.galho.contexto(handle, atuais, vetor)
    ambiguos = []
    for v in atuais:
        r = I.interpretar(v, vocab)
        t = r["tipo"]
        if t == I.CAMINHO_VALIDO:
            colocar("MANTER", v, r["ui"], t, caminho_humano=r["caminho_humano"])
        elif t == I.CAMINHO_ANTIGO:
            colocar("ATUALIZAR", v, r["ui"], t, caminho_humano=r["caminho_humano"],
                    motivo=r["nota"])
        elif t == I.NOME:
            colocar("EXPANDIR", v, r["ui"], t, motivo=(r["nota"] or
                    ("qualificador '%s' guardado no [DECS-ORIGINAL]" % r["qualificador"]
                     if r["qualificador"] else "")))
        elif t in (I.SINONIMO, I.COMBINACAO):
            colocar("ATUALIZAR", v, r["ui"], t,
                    motivo=r["nota"] or "termo de entrada (sinonimo) de '%s'" % vocab.nome(r["ui"]))
        elif t == I.AMBIGUO:
            ambiguos.append((v, r))
        elif t == I.SEM_ARVORE:
            linha("FORA_DA_ARVORE", v, v, r["ui"], t, "", "", r["nota"])
            final.append(v)
        elif t == I.QUALIFICADOR_SOLTO:
            linha("RETIRAR", v, "", "", t, "", "", r["nota"] + " (guardado no [DECS-ORIGINAL])")
        else:
            if config.DECS_TERMO_FORA_DO_DECS == "palavra_chave":
                kws = {norm(x.get("v", "")) for x in item.get("keywords", [])}
                ja = norm(v) in kws
                if not ja:
                    novas_kw.append(v)
                linha("PARA_PALAVRA_CHAVE", v, "" if ja else v, "", t, "", "",
                      "nao e descritor DeCS" + (" (ja estava nas palavras-chave)" if ja else ""))
            else:
                linha("REVISAR", v, v, "", t, "", "", "nao e descritor DeCS")
                final.append(v)

    # Uma chamada so ao modelo: candidatos novos + opcoes dos termos ambiguos
    pontos = []
    if vetor is not None and ctx.ix.modelo is not None:
        extras = {u for _v, r in ambiguos for u in r["candidatos"]}
        pontos = ctx.ix.pontuar(item, vetor, excluir={handle}, extras=extras, proprio=handle)
    p = dict(pontos)

    # ---- 2. ambiguos: o documento decide, com margem ------------------------
    for v, r in ambiguos:
        cands = [u for u in r["candidatos"] if vocab.caminhos(u)]
        escolhido = None
        if p and cands:
            ordem = sorted(cands, key=lambda u: -p.get(u, 0.0))
            p1 = p.get(ordem[0], 0.0)
            p2 = p.get(ordem[1], 0.0) if len(ordem) > 1 else 0.0
            if p1 >= ctx.lim_revisao and p1 - p2 >= MARGEM_DESAMBIGUAR:
                escolhido = (ordem[0], p1)
        if escolhido:
            colocar("DESAMBIGUAR", v, escolhido[0], I.AMBIGUO, prob=escolhido[1],
                    motivo="'%s' serve a %d descritores; o documento indica este"
                    % (v, len(cands)))
        else:
            linha("REVISAR", v, v, "", I.AMBIGUO, "", "",
                  "ambiguo entre: " + "; ".join(vocab.nome(u) for u in cands[:5]))
            final.append(v)

    # ---- 2b. palavra-chave do autor que JA e descritor DeCS (nome ou sinonimo exato) entra no DeCS
    # (regra do responsavel, 2026-10-20). A palavra-chave continua onde esta; o galho sai do modelo de galho.
    from .decs_sinonimos_traicoeiros import traicoeiro
    # o que ja saiu deste item (retirado por nos ou por um catalogador) nao volta (pedido do responsavel, 2026-10-21)
    retirados = R.descritores_retirados(handle, list(item.get("decs") or []))
    bloqueados = {norm(n) for n in config.DECS_CLASSIFICADOR_BLOQUEADOS}
    for k in item.get("keywords") or []:
        if traicoeiro(k.get("v", "")):        # casa com o DeCS, mas quase sempre com outro sentido
            continue
        r = I.interpretar(k.get("v", ""), vocab)
        if (r["tipo"] in (I.NOME, I.SINONIMO) and r["ui"] and r["ui"] not in uis_final and vocab.caminhos(r["ui"])
                and norm(vocab.nome(r["ui"])) not in retirados):
            colocar("ADICIONAR", "", r["ui"], "palavra_chave_autor",
                    motivo="palavra-chave do autor '%s' e descritor DeCS" % k.get("v", ""))

    # ---- 3. descritores novos ---------------------------------------------
    if ctx.sugerir and vetor is not None:
        from . import decs_raros as RR
        raros_llm = []
        n_auto = 0
        for ui, prob in pontos:
            if ui in uis_final or prob < ctx.lim_revisao or norm(vocab.nome(ui)) in retirados | bloqueados:
                continue
            if (config.DECS_LLM_RAROS and prob >= ctx.lim_auto and n_auto < config.DECS_MAX_NOVOS_POR_ITEM
                    and ctx.ix.freq.get(ui, 0) < RR.K_LLM):
                raros_llm.append((ui, prob))           # o LLM decide abaixo (passo 3 dos raros)
                n_auto += 1
                continue
            if prob >= ctx.lim_auto and n_auto < config.DECS_MAX_NOVOS_POR_ITEM:
                n_auto += 1
                colocar("ADICIONAR", "", ui, "classificador", prob=prob,
                        motivo="probabilidade %.2f >= limiar automatico %.2f" % (prob, ctx.lim_auto))
            else:
                caminho, como = ctx.galho.escolher(ui, excluir_handle=handle, contexto=ctx_galho)
                if caminho:
                    linha("SUGERIR", "", caminho, ui, "classificador", como, prob,
                          "para revisao: probabilidade %.2f" % prob)
        if raros_llm:
            from . import decs_segunda_opiniao as SO, ods_professor as P
            if not P.servidor_ativo():
                P.iniciar_servidor()
            ok = SO.confirmar(item, raros_llm, vocab) or []      # falha do LLM = nada confirmado (na duvida, revisa)
            for ui, prob in raros_llm:
                if ui in ok:
                    colocar("ADICIONAR", "", ui, "classificador+llm", prob=prob,
                            motivo="descritor raro (%d exemplos): o LLM confirmou" % ctx.ix.freq.get(ui, 0))
                else:
                    caminho, como = ctx.galho.escolher(ui, excluir_handle=handle, contexto=ctx_galho)
                    if caminho:
                        linha("SUGERIR", "", caminho, ui, "classificador", como, prob,
                              "descritor raro (%d exemplos): o LLM nao confirmou; para revisao"
                              % ctx.ix.freq.get(ui, 0))

    return linhas, final, novas_kw


# ---------------------------------------------------------------------------
# Relatorios
# ---------------------------------------------------------------------------
class Relatorio:
    def __init__(self, rotulo):
        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.caminho = os.path.join(config.SAIDAS_DECS, "plano_%s_%s.csv" % (rotulo, carimbo))
        self.f = open(self.caminho, "w", newline="", encoding="utf-8-sig")
        self.w = csv.DictWriter(self.f, fieldnames=CAMPOS_PLANO, delimiter=";", extrasaction="ignore")
        self.w.writeheader()
        self.contagem = Counter()
        self.itens = 0
        self.itens_com_mudanca = 0

    def escrever(self, linhas, mudou):
        self.itens += 1
        self.itens_com_mudanca += 1 if mudou else 0
        for l in linhas:
            self.w.writerow(l)
            self.contagem[l["acao"]] += 1

    def fechar(self):
        self.f.close()
        return self.caminho


def mudou(item, final, novas_kw):
    return list(item.get("decs", [])) != final or bool(novas_kw)


# ---------------------------------------------------------------------------
# Aplicacao (com todas as travas)
# ---------------------------------------------------------------------------
def item_do_dspace(obj):
    """Objeto de item da API -> dict no formato do acervo (mesmo do D4)."""
    from .decs_coleta import extrair
    return extrair(obj)


def aplicar(cliente, handle, ctx, log_csv=None):
    """Planeja sobre o item LIDO AGORA do servidor e grava. Devolve status.

    Le de novo antes de gravar: o acervo coletado pode estar desatualizado,
    e decidir em cima de dado velho sobrescreveria uma edicao humana recente.
    """
    obj = cliente.item_por_handle(handle)
    if not obj or not obj.get("uuid"):
        return "ERRO_ITEM_NAO_ENCONTRADO", [], []
    handle = obj.get("handle") or handle      # o colado pode ser URL ou UUID
    item = item_do_dspace(obj)
    linhas, final, novas_kw = planejar(handle, item, ctx)
    if not mudou(item, final, novas_kw):
        return "SEM_ALTERACAO", linhas, final

    md = obj.get("metadata") or {}
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    ops = []
    # backup do humano: uma vez so por item
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(handle)
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else atuais)))
    cont = Counter()
    for l in linhas:
        if l["acao"] in ("EXPANDIR", "ATUALIZAR", "DESAMBIGUAR"):
            cont["corrigidos"] += 1
        elif l["acao"] == "ADICIONAR":
            cont["adicionados"] += 1
        elif l["acao"] in ("RETIRAR", "PARA_PALAVRA_CHAVE"):
            cont["removidos"] += 1
    probs = [float(l["probabilidade"]) for l in linhas if l["acao"] == "ADICIONAR"]
    texto_auto = R.linha_auto(cont, len(final), sum(probs) / len(probs) if probs else None)
    ops.extend(R.ops_remover_linhas(md, (R.PREFIXO_AUTO,)))
    if texto_auto:
        ops.append(R.op_adicionar_linha(texto_auto))
    if atuais:
        ops.append({"op": "remove", "path": "/metadata/%s" % config.CAMPO_DECS})
    for v in final:
        ops.append({"op": "add", "path": "/metadata/%s/-" % config.CAMPO_DECS,
                    "value": {"value": v, "language": "pt-BR"}})
    for kw in novas_kw:
        ops.append({"op": "add", "path": "/metadata/%s/-" % config.CAMPO_PALAVRA_CHAVE,
                    "value": {"value": kw, "language": "pt-BR"}})

    cliente.patch("/core/items/" + obj["uuid"], ops)

    # conferencia: o que esta no servidor agora e o que mandamos?
    depois = cliente.get("/core/items/" + obj["uuid"])
    gravado = [x.get("value", "") for x in (depois.get("metadata") or {}).get(config.CAMPO_DECS) or []]
    status = "OK" if gravado == final else "DIVERGENTE"
    R.registrar(handle, atuais, gravado, linhas, aplicado=True,
                confianca=(sum(probs) / len(probs)) if probs else None, limiar=ctx.lim_auto)
    return status, linhas, final


def restaurar(cliente, handle, confirmar):
    """Volta o dc.subject.decs ao que o catalogador tinha ([DECS-ORIGINAL])."""
    obj = cliente.item_por_handle(handle)
    if not obj:
        return "ERRO_ITEM_NAO_ENCONTRADO", None, None
    handle = obj.get("handle") or handle
    md = obj.get("metadata") or {}
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    original, fonte = R.ler_original(md), "provenance do item ([DECS-ORIGINAL])"
    if original is None:
        original, fonte = R.humano_antes_da_primeira_aplicacao(handle), "banco local (SQLite)"
    if original is None:
        return "SEM_BACKUP", atuais, None
    if original == atuais:
        return "JA_ORIGINAL", atuais, original
    if not confirmar(atuais, original, fonte):
        return "CANCELADO", atuais, original
    ops = R.ops_remover_linhas(md, (R.PREFIXO_AUTO, R.PREFIXO_RESTAURADO))
    ops.append(R.op_adicionar_linha(R.linha_restaurado(len(original), fonte)))
    if atuais:
        ops.append({"op": "remove", "path": "/metadata/%s" % config.CAMPO_DECS})
    for v in original:
        ops.append({"op": "add", "path": "/metadata/%s/-" % config.CAMPO_DECS,
                    "value": {"value": v, "language": "pt-BR"}})
    cliente.patch("/core/items/" + obj["uuid"], ops)
    R.registrar(handle, atuais, original, [{"acao": "RESTAURAR", "fonte": fonte}], aplicado=True)
    return "OK", atuais, original


def gravar_novos(cliente, handle, linhas, ctx, acrescentar=False):
    """Grava descritores NOVOS aprovados pelo usuario. Le o item agora antes de gravar.

    `linhas`: linhas ADICIONAR/SUGERIR do planejar(). Sem `acrescentar`, so grava em item SEM DeCS (se um
    catalogador indexou desde a consulta, nao grava). Com `acrescentar`, junta aos descritores que o item ja tem,
    sem tocar neles e sem repetir. [DECS-ORIGINAL] guarda o humano antes da 1a gravacao (regra 4) e [DECS-AUTO]
    marca a origem: saida do modelo nunca vira exemplo de treino (regra 7).
    """
    obj = cliente.item_por_handle(handle)
    if not obj or not obj.get("uuid"):
        return "ERRO_ITEM_NAO_ENCONTRADO"
    md = obj.get("metadata") or {}
    atuais = [x.get("value", "") for x in md.get(config.CAMPO_DECS) or []]
    if atuais and not acrescentar:
        return "TEM_DECS_AGORA (nada gravado)"
    caminhos = [c for c in dict.fromkeys(l["depois"] for l in linhas if l.get("depois")) if c not in atuais]
    if not caminhos:
        return "SEM_ALTERACAO"
    probs = [float(l["probabilidade"]) for l in linhas if l.get("probabilidade")]
    ops = []
    if R.ler_original(md) is None:
        humano = R.humano_antes_da_primeira_aplicacao(obj.get("handle") or handle) if atuais else None
        ops.append(R.op_adicionar_linha(R.linha_original(humano if humano is not None else atuais)))
    texto_auto = R.linha_auto(Counter({"adicionados": len(caminhos)}), len(atuais) + len(caminhos),
                              sum(probs) / len(probs) if probs else None)
    ops.extend(R.ops_remover_linhas(md, (R.PREFIXO_AUTO,)))
    if texto_auto:
        ops.append(R.op_adicionar_linha(texto_auto))
    for v in caminhos:
        ops.append({"op": "add", "path": "/metadata/%s/-" % config.CAMPO_DECS,
                    "value": {"value": v, "language": "pt-BR"}})
    cliente.patch("/core/items/" + obj["uuid"], ops)
    depois = cliente.get("/core/items/" + obj["uuid"])
    gravado = [x.get("value", "") for x in (depois.get("metadata") or {}).get(config.CAMPO_DECS) or []]
    R.registrar(obj.get("handle") or handle, atuais, gravado,
                [dict(l, acao="ADICIONAR") for l in linhas], aplicado=True,
                confianca=(sum(probs) / len(probs)) if probs else None, limiar=ctx.lim_revisao)
    return "OK" if gravado == atuais + caminhos else "DIVERGENTE"
