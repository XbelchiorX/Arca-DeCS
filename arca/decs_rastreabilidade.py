# -*- coding: utf-8 -*-
"""Rastreabilidade: o que o script fez, em que item, com que modelo.

DUAS CAMADAS
1. No proprio item, em dcterms.provenance (NAO sai no OAI: oai_dc, xoai,
   mets e dim conferidos em 2026-09-30). Tres tipos de linha, todas com
   prefixo, para o script so mexer nas proprias:
     [DECS-ORIGINAL] os descritores que o CATALOGADOR tinha. Gravada UMA
                     vez por item, antes da primeira alteracao. E o backup
                     que viaja com o item e permite restaurar (opcao D0).
     [DECS-AUTO]     resumo da ultima indexacao assistida (substituida a cada
                     aplicacao).
     [DECS-RESTAURADO] aviso de que o original foi devolvido.
   dc.description.provenance (workflow do DSpace) NUNCA e tocado.

2. No banco local dados/decs_indexacao.db (SQLite): o detalhe completo de
   cada decisao — acao, valor antes/depois, metodo, probabilidade, motivo,
   versao do modelo — inclusive das simulacoes (coluna `aplicado` = 0).
"""
import json
import os
import sqlite3
from datetime import datetime

from . import config

PREFIXO_ORIGINAL = "[DECS-ORIGINAL]"
PREFIXO_AUTO = "[DECS-AUTO]"
PREFIXO_RESTAURADO = "[DECS-RESTAURADO]"
# ODS: mesmas regras, prefixos proprios
PREFIXO_ODS_ORIGINAL = "[ODS-ORIGINAL]"
PREFIXO_ODS_AUTO = "[ODS-AUTO]"
PREFIXO_ODS_RESTAURADO = "[ODS-RESTAURADO]"
SEP_LISTA = " >>> "
CAMPO = "dcterms.provenance"


# ---------------------------------------------------------------------------
# Linhas de provenance
# ---------------------------------------------------------------------------
def linha_original(valores, prefixo=PREFIXO_ORIGINAL):
    """'[DECS-ORIGINAL] 2026-10-02 14:32 (v9.0.0) >>> ["A", "B"]'.

    JSON e nao ' | ': um valor com barra vertical voltaria partido. '[]' quer
    dizer "o item nao tinha descritor" (restaurar para vazio e valido) — e
    diferente de nao haver linha.
    """
    lista = [str(x).strip() for x in valores if str(x).strip()]
    return "%s %s (v%s)%s%s" % (prefixo, datetime.now().strftime("%Y-%m-%d %H:%M"),
                                config.VERSAO, SEP_LISTA, json.dumps(lista, ensure_ascii=False))


def ler_original(metadata, prefixo=PREFIXO_ORIGINAL):
    """Lista gravada em [DECS-ORIGINAL] (ou outro prefixo), ou None se nao houver."""
    for v in metadata.get(CAMPO) or []:
        txt = v.get("value") or ""
        if txt.startswith(prefixo) and SEP_LISTA in txt:
            try:
                lista = json.loads(txt.split(SEP_LISTA, 1)[1].strip())
            except ValueError:
                return None     # linha corrompida: melhor "nao sei" do que "vazio"
            return [str(x) for x in lista] if isinstance(lista, list) else None
    return None


def linha_auto(contagem, n_final, confianca):
    partes = []
    for chave, rotulo in (("corrigidos", "corrigido(s)"), ("adicionados", "adicionado(s)"),
                          ("removidos", "retirado(s) do campo")):
        if contagem.get(chave):
            partes.append("%d %s" % (contagem[chave], rotulo))
    if not partes:
        return ""
    conf = (" Confianca media dos novos: %.2f." % confianca) if confianca is not None else ""
    return ("%s Indexacao DECS assistida por IA - %s. %s; %d descritor(es) no total. "
            "Modelo %s (script v%s).%s Detalhe na base interna de indexacao da Fiocruz."
            % (PREFIXO_AUTO, datetime.now().strftime("%Y-%m-%d"), "; ".join(partes),
               n_final, config.VERSAO_MODELO, config.VERSAO, conf))


def linha_restaurado(n, fonte):
    return "%s %s (v%s) - %d descritor(es) restaurado(s) de: %s." % (
        PREFIXO_RESTAURADO, datetime.now().strftime("%Y-%m-%d %H:%M"), config.VERSAO, n, fonte)


def ops_remover_linhas(metadata, prefixos):
    """Remove SO as linhas do script, por indice, do maior para o menor.

    Nunca {"op": "remove", "path": "/metadata/dcterms.provenance"}: apaga o
    campo inteiro (o cliente tambem bloqueia).
    """
    valores = [x.get("value", "") for x in (metadata.get(CAMPO) or [])]
    idx = [i for i, v in enumerate(valores) if v.startswith(tuple(prefixos))]
    return [{"op": "remove", "path": "/metadata/%s/%d" % (CAMPO, i)} for i in sorted(idx, reverse=True)]


def op_adicionar_linha(texto):
    return {"op": "add", "path": "/metadata/%s/-" % CAMPO, "value": {"value": texto, "language": "en"}}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
def _conectar():
    db = sqlite3.connect(config.BANCO_SQLITE, timeout=60)   # varias threads gravando: espera em vez de "database is locked"
    db.execute("""CREATE TABLE IF NOT EXISTS indexacao_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, handle TEXT NOT NULL,
        data TIMESTAMP DEFAULT CURRENT_TIMESTAMP, modelo TEXT NOT NULL,
        versao_script TEXT NOT NULL, decs_antes TEXT, decs_depois TEXT,
        mudancas_detalhadas TEXT, confianca_media REAL, hash_embeddings TEXT,
        threshold_usado REAL, trilha_idioma TEXT, reprocessado INTEGER DEFAULT 0,
        data_reprocessado TIMESTAMP, modelo_reprocessado TEXT)""")
    cols = {r[1] for r in db.execute("PRAGMA table_info(indexacao_log)")}
    if "aplicado" not in cols:
        # registros anteriores a v9: o script antigo gravava simulacoes e
        # aplicacoes juntas; ficam como NULL ("nao se sabe").
        db.execute("ALTER TABLE indexacao_log ADD COLUMN aplicado INTEGER")
    db.execute("CREATE INDEX IF NOT EXISTS idx_handle ON indexacao_log(handle)")
    return db


def registrar(handle, antes, depois, linhas, aplicado, confianca=None, limiar=None):
    with _conectar() as db:
        if aplicado:
            db.execute("""UPDATE indexacao_log SET reprocessado=1, data_reprocessado=?,
                          modelo_reprocessado=? WHERE handle=? AND reprocessado=0
                          AND modelo<>? AND aplicado<>0""",
                       (datetime.now().isoformat(), config.VERSAO_MODELO, handle,
                        config.VERSAO_MODELO))
        db.execute("""INSERT INTO indexacao_log (handle, modelo, versao_script, decs_antes,
                      decs_depois, mudancas_detalhadas, confianca_media, threshold_usado,
                      aplicado) VALUES (?,?,?,?,?,?,?,?,?)""",
                   (handle, config.VERSAO_MODELO, config.VERSAO,
                    json.dumps(antes, ensure_ascii=False), json.dumps(depois, ensure_ascii=False),
                    json.dumps(linhas, ensure_ascii=False), confianca, limiar,
                    1 if aplicado else 0))


def humano_antes_da_primeira_aplicacao(handle):
    """decs_antes da PRIMEIRA vez que o script gravou no item (None se nunca gravou).

    Registros antigos (aplicado NULL) contam: o script antigo nao distinguia,
    e o decs_antes do primeiro registro e sempre o estado humano.
    """
    if not os.path.exists(config.BANCO_SQLITE):
        return None
    with _conectar() as db:
        r = db.execute("""SELECT decs_antes FROM indexacao_log WHERE handle=?
                          AND (aplicado IS NULL OR aplicado=1) ORDER BY id LIMIT 1""",
                       (handle,)).fetchone()
    try:
        lista = json.loads(r[0]) if r and r[0] else None
    except ValueError:
        return None
    return [str(x) for x in lista] if isinstance(lista, list) else None


def ultima_saida(handle):
    """decs_depois da ultima gravacao do script no item (None se nao ha registro)."""
    if not os.path.exists(config.BANCO_SQLITE):
        return None
    with _conectar() as db:
        r = db.execute("""SELECT decs_depois FROM indexacao_log WHERE handle=?
                          AND (aplicado IS NULL OR aplicado=1) ORDER BY id DESC LIMIT 1""",
                       (handle,)).fetchone()
    try:
        lista = json.loads(r[0]) if r and r[0] else None
    except ValueError:
        return None
    return [str(x) for x in lista] if isinstance(lista, list) else None


def descritores_retirados(handle, atuais=None):
    """Nomes de descritor (o ultimo no do caminho, normalizado) que ja SAIRAM deste item e nao devem voltar
    (pedido do responsavel, 2026-10-21: o D2 rodado de novo recolocava o que foi retirado).

    Conta duas coisas: o que uma gravacao registrada tirou (estava em decs_antes e nenhum caminho com o mesmo
    descritor ficou em decs_depois; uma troca de galho ou de raiz nao conta) e o que alguem tirou direto no DSpace
    depois da ultima gravacao do script (estava na ultima saida e nao esta em `atuais`)."""
    from .decs_vocabulario import norm
    if not os.path.exists(config.BANCO_SQLITE):
        return set()
    folha = lambda v: norm(str(v).split("::")[-1])
    saiu = set()
    ultima = None
    with _conectar() as db:
        for antes, depois in db.execute("""SELECT decs_antes, decs_depois FROM indexacao_log WHERE handle=?
                                           AND aplicado=1 ORDER BY id""", (handle,)):
            try:
                antes, depois = json.loads(antes or "[]"), json.loads(depois or "[]")
            except ValueError:
                continue
            ficou = {folha(v) for v in depois}
            saiu |= {folha(v) for v in antes} - ficou
            ultima = depois
    if atuais is not None and ultima is not None:
        saiu |= {folha(v) for v in ultima} - {folha(v) for v in atuais}
    if atuais is not None:
        saiu -= {folha(v) for v in atuais}      # o que esta no item agora nunca conta como retirado
    return saiu


def estatisticas():
    with _conectar() as db:
        return db.execute("""SELECT modelo, COALESCE(aplicado, -1), COUNT(*), MIN(data), MAX(data)
                             FROM indexacao_log GROUP BY modelo, aplicado ORDER BY MAX(data)""").fetchall()


def reprocessaveis():
    """Handles cuja ultima APLICACAO usou um modelo diferente do atual."""
    with _conectar() as db:
        linhas = db.execute("""SELECT handle, modelo FROM indexacao_log WHERE id IN (
            SELECT MAX(id) FROM indexacao_log WHERE aplicado IS NULL OR aplicado=1
            GROUP BY handle)""").fetchall()
    return sorted(h for h, m in linhas if m != config.VERSAO_MODELO)
