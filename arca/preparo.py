# -*- coding: utf-8 -*-
"""Confere, antes de cada fluxo do DeCS, se os dados locais existem, e resolve sozinho o que faltar.

Pedido do responsavel (2026-10-19): a suite tem de ser autonoma, prever o erro e resolver. Caso real: o D1 parou
com "Acervo nao coletado" porque a copia local, os modelos e os embeddings so existiam noutro ambiente.

Ordem de solucao, para cada arquivo que falta (nada disso grava no ARCA):
1. copia do espelho no NAS (ARCA_DADOS_NAS; o mesmo arquivo, em minutos);
2. refaz: coleta do acervo (D4, so leitura), vocabulario do servidor (D6, so leitura), treino do modelo (D5);
   embeddings e modelo de galho se refazem sozinhos quando o programa os pede.
So para quando nao ha como resolver sem a pessoa (ex.: o XML da BIREME, que vem de download manual).
"""
import os
import shutil
import subprocess
import sys

from . import config

NAS_DADOS = os.getenv("ARCA_DADOS_NAS", "//SEU_NAS/compartilhamento/CuradoriaARCA/transferencia")

# (descricao, caminho relativo a DADOS_LOCAIS ou absoluto, solucao quando nao ha copia no NAS)
NECESSARIOS_DECS = [
    ("Copia local do acervo (base dos modelos)", "acervo_arca.json", "coletar"),
    ("Vocabulario decs publicado no servidor", config.VOCAB_SERVIDOR_JSON, "vocabulario"),
    ("Base DeCS da BIREME (XML)", config.BIREME_XML_POR, "manual"),
    ("Modelo de sugestao de DeCS", "modelo_decs.pkl", "treinar"),
    ("Rede multirrotulo do modelo de sugestao", "rede_decs.pt", "treinar"),
    ("Embeddings do acervo (e5-large)", os.path.join("cache", "emb_itens_e5-large.pt"), "sozinho"),
    ("Embeddings do acervo (Qwen3-8B, galhos)", os.path.join("cache", "emb_itens_qwen3-8b.pt"), "emb_qwen"),
    ("Modelo de galho (Qwen3-8B)", "modelo_galho_soltos_qwen3-8b.pkl", "sozinho"),
    ("Textos dos galhos (Qwen3-8B)", os.path.join("cache", "texto_galhos_qwen3-8b.pt"), "sozinho"),
]


def _abs(caminho):
    return caminho if os.path.isabs(caminho) else os.path.join(config.DADOS_LOCAIS, caminho)


def _do_nas(rel, aviso):
    if os.path.isabs(rel):
        return False
    origem = os.path.join(NAS_DADOS, rel)
    if not os.path.isfile(origem):
        return False
    destino = _abs(rel)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    aviso("  copiando do NAS: %s (%.0f MB)" % (rel, os.path.getsize(origem) / 1e6))
    shutil.copy2(origem, destino + ".tmp")
    os.replace(destino + ".tmp", destino)
    return True


def _rodar(script, env_extra=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **(env_extra or {}))
    return subprocess.run([sys.executable, os.path.join(config.RAIZ, "ferramentas", "decs", script)],
                          cwd=config.RAIZ, env=env).returncode == 0


def garantir(necessarios=None, aviso=print):
    """Resolve o que falta. Devolve a lista do que nao foi possivel resolver (vazia = pronto)."""
    necessarios = necessarios or NECESSARIOS_DECS
    falta = [n for n in necessarios if not os.path.exists(_abs(n[1]))]
    if not falta:
        return []
    aviso("Faltam %d arquivo(s) local(is) do DeCS; resolvendo sozinho (nada e gravado no ARCA):" % len(falta))
    for desc, rel, _ in falta:
        aviso("  - %s" % desc)
    restam = []
    for n in falta:
        if not _do_nas(n[1], aviso):
            restam.append(n)
    solucoes = {s for _, _, s in restam}
    if "coletar" in solucoes:
        aviso("  coletando o acervo do ARCA (so leitura; ~20 min)...")
        from . import dspace, decs_coleta
        decs_coleta.coletar(dspace.conectar(aviso=aviso), progresso=aviso)
    if "vocabulario" in solucoes:
        aviso("  baixando o vocabulario decs do servidor (so leitura)...")
        _rodar("baixar_vocabulario_servidor.py")
    if "treinar" in solucoes:
        aviso("  treinando e avaliando o modelo de sugestao (pode levar mais de uma hora)...")
        _rodar("treinar_avaliar.py")
    if "emb_qwen" in solucoes:
        aviso("  gerando os embeddings Qwen3-8B do acervo (muitas horas na primeira vez; retoma de onde parou)...")
        subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); from arca import decs_acervo, "
                        "decs_indexador as IX; IX.embeddings_itens(decs_acervo.carregar_acervo())" % config.RAIZ],
                       cwd=config.RAIZ, env=dict(os.environ, DECS_MODELO_EMBEDDING="qwen3-8b", PYTHONIOENCODING="utf-8"))
    nao = [n for n in falta if n[2] != "sozinho" and not os.path.exists(_abs(n[1]))]
    for desc, rel, sol in nao:
        if sol == "manual":
            aviso("  NAO resolvido: %s. Baixe o XML do DeCS 2026 da BIREME e ponha em %s." % (desc, _abs(rel)))
        else:
            aviso("  NAO resolvido: %s (%s)." % (desc, _abs(rel)))
    return nao
