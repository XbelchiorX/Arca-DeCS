# -*- coding: utf-8 -*-
"""Ajuste e correcao do atributo `language` dos metadados (opcoes 1-9, A, B).

ESTE MODULO FOI MOVIDO DO SCRIPT v8.2.1 (extraido por analise de dependencias
em 2026-10-02). Mudou a "casca" e, depois de medido, a DECISAO de idioma:
  - a deteccao usa arca/idioma_deteccao.py e a decisao de dc.title,
    dc.title.alternative e dc.description passa por decidir_idioma()
    (acerto nas discordancias: 35% -> 95%; ver docs/DECISOES.md secao 8);
  - login e HTTP passam pelo cliente unico de arca/dspace.py, que tem as
    travas (somente-leitura, provenance do DSpace intocavel);
  - pastas vem de arca/config.py;
  - o relatorio CSV so e criado quando ha a primeira alteracao (antes, cada
    vez que o script era aberto nascia um CSV vazio: eram 451 na pasta).

O que faz:
  dc.title, dc.title.alternative, dc.description: normaliza language para
  BCP 47 (pt_BR -> pt-BR) e, se vazio (LIMPAR) ou tambem se errado
  (CONSERTAR), detecta pelo texto com arca/idioma_deteccao.py; quando o
  detector nao tem seguranca, mantem o idioma gravado.
  Campos de idioma fixo (dc.subject.mesh = en, dc.description.abstracten = en...).
  Demais metadados: language = None.
  dc.description.provenance e dcterms.provenance: somente leitura.
"""
import concurrent.futures
import csv
import os
import re
import threading
import time
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urlparse

from rich import box
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from . import config
from . import dspace as _dspace
from . import idioma_deteccao as _deteccao
from .ui import aviso, console, erro, info, ok, titulo

try:
    from lingua import Language, LanguageDetectorBuilder

    _IDIOMAS_DETECCAO = [
        Language.PORTUGUESE, Language.ENGLISH, Language.SPANISH, Language.FRENCH,
        Language.GERMAN, Language.ITALIAN, Language.JAPANESE, Language.CHINESE,
        Language.KOREAN, Language.RUSSIAN, Language.ARABIC, Language.TURKISH,
    ]
    _DETECTOR_LINGUA = (LanguageDetectorBuilder.from_languages(*_IDIOMAS_DETECCAO)
                        .with_minimum_relative_distance(0.0).build())
    LINGUA_DISPONIVEL = True
except ImportError:
    _DETECTOR_LINGUA = None
    LINGUA_DISPONIVEL = False
DETECCAO_DISPONIVEL = LINGUA_DISPONIVEL

SAIDA_IDIOMAS = config.SAIDAS_IDIOMAS
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Casca de compatibilidade com o cliente novo (arca/dspace.py)
# ---------------------------------------------------------------------------
ErroHTTPDSpace = _dspace.ErroDSpace


class _ClienteCompat:
    """Expoe a interface antiga (get_json/patch_json/base_url) sobre o cliente novo."""

    def __init__(self, cliente):
        self._c = cliente
        self.base_url = cliente.base

    def login(self):
        pass                       # o cliente novo ja nasce autenticado

    def get_json(self, url, params=None, required=True):
        return self._c.get(url, params=params, obrigatorio=required)

    def patch_json(self, url, operacoes, required=True):
        return self._c.patch(url, operacoes)


def obter_cookies_cdp():
    return None                    # a sessao e gerida por arca/dspace.py


def criar_cliente(quiet=False, cdp_cookies=None):
    return _ClienteCompat(_dspace.conectar(aviso=info))


def get_thread_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = _ClienteCompat(_dspace.conectar(aviso=info))
    return _thread_local.client


class RelatorioCSV:
    """CSV de alteracoes de idioma; o arquivo so nasce na primeira linha."""

    def __init__(self):
        self.caminho = os.path.join(
            SAIDA_IDIOMAS, "alteracoes_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.lock = threading.Lock()
        self.total_linhas = 0

    def registrar(self, uuid, campo, indice, anterior, novo, motivo):
        with self.lock:
            novo_arquivo = not os.path.exists(self.caminho)
            with open(self.caminho, "a", newline="", encoding="utf-8-sig") as arq:
                w = csv.writer(arq, delimiter=";")
                if novo_arquivo:
                    w.writerow(["uuid", "campo", "indice", "valor_anterior", "valor_novo", "motivo"])
                w.writerow([uuid, campo, indice, anterior if anterior is not None else "",
                            novo if novo is not None else "", motivo])
            self.total_linhas += 1


RELATORIO = RelatorioCSV()


CAMPOS_PRESERVAR_IDIOMA = {
    "dc.title",
    "dc.description",
}


CAMPOS_IDIOMA_FIXO_PT = {
    "dc.subject.decs",
    "dc.subject.ods",
}


CAMPOS_IDIOMA_FIXO = {
    "dc.subject.other": "pt-BR",
    "dc.subject.en": "en",
    "dc.subject.fr": "fr",
    "dc.subject.es": "es",
    "dc.subject.mesh": "en",
    "dc.description.abstract": "pt-BR",
    "dc.description.abstracten": "en",
    "dc.description.abstractfr": "fr",
    "dc.description.abstractes": "es",
}


CAMPO_CONTRIBUTOR_AUTHOR = "dc.contributor.author"


CAMPO_TITLE = "dc.title"


CAMPO_TITLE_ALT = "dc.title.alternative"


# ----------------------------------------------------------------------
# PROVENANCE: DUAS COISAS COMPLETAMENTE DIFERENTES
# ----------------------------------------------------------------------
# 1) O QUE O DSPACE ESCREVE — intocavel.
#    "Submitted by ...", "Step: reviewstep - action:reviewaction",
#    "Made available in DSpace ...". E o registro do workflow, escrito pelo
#    proprio DSpace e pelas pessoas. Este script NAO escreve, NAO remove e NAO
#    normaliza aqui. Nem no caminho feliz, nem por engano, nem em bug.
#
#    Em 2026-09-30 um `remove` do campo inteiro apagou as 4 linhas de workflow
#    do item icict/78513 (elas nao estao no OAI nem no historico de versoes,
#    entao nao voltaram). E o motivo de a trava existir.
CAMPO_PROVENANCE_DSPACE = "dc.description.provenance"


# 2) ONDE O SCRIPT GRAVA O QUE E DELE.
#    dcterms.provenance = "alteracoes desde a criacao". Semanticamente e o
#    registro do que o humano tinha antes de nos mexermos, que e exatamente o
#    que a linha [DECS-ORIGINAL] guarda.
#
#    Medido nesta instancia: o campo existe, aceita escrita, o workflow nao o
#    usa, e nao aparece em nenhum formato OAI (oai_dc, xoai, mets, dim).
#    Se algum dia o DSpace passar a usa-lo, CAMPO_PROVENANCE_TROCA aponta a
#    alternativa (dc.provenance, tambem verificada sem vazamento no OAI).
CAMPO_PROVENANCE = "dcterms.provenance"


# Campos que o script pode ler mas nunca escrever.
CAMPOS_PROIBIDOS_ESCRITA = (CAMPO_PROVENANCE_DSPACE,)


CAMPOS_SOMENTE_LEITURA = (CAMPO_PROVENANCE_DSPACE, CAMPO_PROVENANCE)


DEFAULT_PAGE_SIZE = 50


MAX_SAFE_PAGE_SIZE = 8


DEFAULT_MAPPING_WORKERS = 12


DEFAULT_PROCESS_WORKERS = 5


PATCH_BATCH_SIZE = 50


ITEM_MAX_TENTATIVAS = 3


DETECCAO_MINIMO_CHARS = 12


LIMIAR_CONFIANCA_ALTA = 0.90


Pares_LINGUA_AMBIGUA = {
    frozenset({"pt-BR", "es"}),
    frozenset({"pt-BR", "it"}),
    frozenset({"pt-BR", "fr"}),
    frozenset({"es", "it"}),
    frozenset({"es", "fr"}),
    frozenset({"it", "fr"}),
    frozenset({"en", "nl"}),
    frozenset({"en", "de"}),
    frozenset({"en", "sv"}),
    frozenset({"no", "da", "sv"}),
}


# =============================================================================
# PAGINAÇÃO E RESOLUÇÃO
# =============================================================================
def get_embedded_list(data, nome):
    embedded = (data or {}).get("_embedded", {})
    valor = embedded.get(nome)
    if isinstance(valor, list):
        return valor
    if isinstance(valor, dict):
        interno = valor.get("_embedded", {}).get(nome)
        if isinstance(interno, list):
            return interno
    return []


def get_paged_parallel(cliente, url, embedded_key, page_size, workers, params=None):
    """Mapeamento paginado e paralelo UNICO da suite (opcoes de idiomas, LICENSE L2/L3 e thumbnails M).
    `params` extras (ex.: {"embed": "bundles"}) vao em todas as paginas."""
    page_size = min(max(1, int(page_size)), MAX_SAFE_PAGE_SIZE)
    workers = max(1, int(workers))
    extras = dict(params or {})
    parametros = dict(extras, size=page_size, page=0)
    primeira = cliente.get_json(url, params=parametros, required=True)
    itens = get_embedded_list(primeira, embedded_key)
    pagina = primeira.get("page") or {}
    total_paginas = int(pagina.get("totalPages", 1) or 1)
    total_elementos = pagina.get("totalElements")

    console.print(
        Panel.fit(
            f"[bold]Paginação detectada[/bold]\n"
            f"Page size: {page_size} | Páginas: {total_paginas} | "
            f"Elementos: {total_elementos}",
            border_style="cyan",
        )
    )
    if total_paginas <= 1:
        return itens

    def buscar(numero):
        cliente_thread = get_thread_client()
        dados = cliente_thread.get_json(url, params=dict(extras, size=page_size, page=numero), required=True)
        return numero, get_embedded_list(dados, embedded_key)

    erros = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progresso:
        tarefa = progresso.add_task("Mapeando páginas", total=total_paginas - 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futuros = {executor.submit(buscar, n): n for n in range(1, total_paginas)}
            for futuro in concurrent.futures.as_completed(futuros):
                numero = futuros[futuro]
                try:
                    _, pagina_itens = futuro.result()
                    itens.extend(pagina_itens)
                except Exception as exc:
                    erros.append((numero, exc))
                progresso.update(tarefa, advance=1)

    if erros:
        detalhes = " | ".join(f"página {n}: {e}" for n, e in erros[:10])
        raise Exception(
            f"Mapeamento incompleto: {len(erros)} página(s) falharam. {detalhes}"
        )
    return itens


def normalizar_identificador(valor):
    valor = str(valor or "").strip()
    if not valor:
        return ""
    if valor.startswith(("http://", "https://")):
        caminho = urlparse(valor).path
        match = re.search(
            r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
            caminho,
        )
        if match:
            return match.group(1)
    return valor


def eh_uuid(valor):
    try:
        uuid.UUID(str(valor))
        return True
    except Exception:
        return False


def resolver_por_handle(cliente, handle):
    """Resolve um handle (ex.: "icict/78513") no item correspondente.

    O jeito certo no DSpace 7/9 e GET /pid/find?id=<handle>. Isto aqui foi
    consertado em 2026-09-30 porque a resolucao por handle estava
    INTEIRAMENTE QUEBRADA nesta instancia e o D1 nao aplicava nada:

      GET /api/core/items/icict/78513              -> 500 (a barra no path)
      GET /api/core/items/find-by-handle           -> 404 (rota inexistente)
      GET /api/core/items/search/byHandle          -> 404 (rota inexistente)
      GET /api/pid/find?id=icict/78513             -> 200  <-- este

    Sintoma: o passo de aplicacao escrevia ERRO_ITEM_NAO_ENCONTRADO em todas as
    linhas do log, imprimia "Aplicação concluída" e nao alterava nada no Arca.
    Como o CSV usa handle e nao uuid, isso atingia TODOS os itens.
    """
    try:
        dado = cliente.get_json(f"{cliente.base_url}/pid/find",
                                params={"id": handle}, required=False)
    except Exception:
        return None
    if isinstance(dado, dict) and dado.get("uuid"):
        return dado
    return None


def resolver_item(cliente, entrada):
    entrada = normalizar_identificador(entrada)
    if eh_uuid(entrada):
        return cliente.get_json(
            f"{cliente.base_url}/core/items/{entrada}", required=False
        )
    if "/" in entrada:
        dado = resolver_por_handle(cliente, entrada)
        if dado:
            return dado
        # formatos antigos de prefixo, ex.: "hdl:arca.fiocruz.br/icict/78513"
        if entrada.startswith("hdl:"):
            sufixo = entrada.split("hdl:", 1)[1].split("/", 1)[-1]
            return resolver_por_handle(cliente, sufixo)
    return None


def resolver_colecao(cliente, entrada):
    entrada = normalizar_identificador(entrada)
    if eh_uuid(entrada):
        return cliente.get_json(
            f"{cliente.base_url}/core/collections/{entrada}", required=False
        )
    if "/" in entrada:
        for endpoint in (
            f"{cliente.base_url}/core/collections/find-by-handle",
            f"{cliente.base_url}/core/collections/search/byHandle",
            f"{cliente.base_url}/pid/find",
        ):
            dado = cliente.get_json(
                endpoint, params={"handle": entrada, "id": entrada}, required=False
            )
            if isinstance(dado, dict) and dado.get("uuid"):
                return dado
    return None


def mapear_objetos(cliente, page_size, workers, data_inicio=None, params=None):
    """Itens do ARCA agora (objetos completos), com o filtro de data opcional. Usado por todas as opcoes."""
    titulo("MAPEAMENTO GLOBAL")
    itens = get_paged_parallel(
        cliente,
        f"{cliente.base_url}/core/items",
        "items",
        page_size,
        workers,
        params=params,
    )
    if data_inicio is not None:
        itens, mantidos, removidos = filtrar_por_data(itens, data_inicio)
        info(f"Filtro de data: {mantidos} itens a partir de {data_inicio} "
             f"(removidos {removidos} anteriores).")
    return itens


def mapear_todos(cliente, page_size, workers, data_inicio=None):
    itens = mapear_objetos(cliente, page_size, workers, data_inicio=data_inicio)
    uuids = [i.get("uuid") for i in itens if i.get("uuid")]
    ok(f"{len(uuids)} item(ns) mapeado(s).")
    return uuids


def mapear_colecao(cliente, entrada, page_size, workers, data_inicio=None):
    titulo("MAPEAMENTO POR COLEÇÃO")
    colecao = resolver_colecao(cliente, entrada)
    if not colecao or not colecao.get("uuid"):
        raise Exception("Coleção não encontrada.")
    href = (
        colecao.get("_links", {}).get("items", {}).get("href")
        or f"{cliente.base_url}/core/collections/{colecao['uuid']}/items"
    )
    itens = get_paged_parallel(cliente, href, "items", page_size, workers)
    if data_inicio is not None:
        itens, mantidos, removidos = filtrar_por_data(itens, data_inicio)
        info(f"Filtro de data: {mantidos} itens a partir de {data_inicio} "
             f"(removidos {removidos} anteriores).")
    return [i.get("uuid") for i in itens if i.get("uuid")]


# =============================================================================
# IDIOMA E PATCH ROBUSTO
# =============================================================================
LANGUAGE_ALIASES = {
    "pt_br": "pt-BR", "pt-br": "pt-BR", "ptbr": "pt-BR", "br": "pt-BR",
    "pt": "pt-BR", "por": "pt-BR", "portuguese": "pt-BR", "portugues": "pt-BR",
    "português": "pt-BR", "en": "en", "eng": "en", "english": "en",
    "ingles": "en", "inglês": "en", "en_us": "en-US", "en-us": "en-US",
    "en_gb": "en-GB", "en-gb": "en-GB", "es": "es", "spa": "es",
    "spanish": "es", "espanhol": "es", "fr": "fr", "fra": "fr",
    "fre": "fr", "french": "fr", "francês": "fr", "de": "de",
    "deu": "de", "ger": "de", "german": "de", "alemão": "de",
    "alemao": "de", "it": "it", "ita": "it", "italian": "it",
    "italiano": "it", "ja": "ja", "jpn": "ja", "japanese": "ja",
    "zh": "zh", "zho": "zh", "chi": "zh", "chinese": "zh",
    "ko": "ko", "kor": "ko", "korean": "ko", "ru": "ru",
    "rus": "ru", "russian": "ru", "ar": "ar", "ara": "ar",
    "arabic": "ar", "tr": "tr", "tur": "tr", "turkish": "tr",
}


LINGUA_PARA_BCP47 = {
    "PORTUGUESE": "pt-BR", "ENGLISH": "en", "SPANISH": "es",
    "FRENCH": "fr", "GERMAN": "de", "ITALIAN": "it",
    "JAPANESE": "ja", "CHINESE": "zh", "KOREAN": "ko",
    "RUSSIAN": "ru", "ARABIC": "ar", "TURKISH": "tr",
}


def normalizar_language(language):
    if language is None:
        return None
    original = str(language).strip()
    if not original:
        return original
    chave = original.lower()
    if chave in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[chave]
    partes = [p for p in original.replace("_", "-").split("-") if p]
    if not partes or not all(re.fullmatch(r"[A-Za-z0-9]{1,8}", p) for p in partes):
        return original
    saida = []
    for indice, parte in enumerate(partes):
        if indice == 0:
            saida.append(parte.lower())
        elif len(parte) == 4 and parte.isalpha():
            saida.append(parte.title())
        elif (len(parte) == 2 and parte.isalpha()) or (
            len(parte) == 3 and parte.isdigit()
        ):
            saida.append(parte.upper())
        else:
            saida.append(parte.lower())
    return "-".join(saida)


def detectar_idioma(texto):
    """Idioma do texto ou None (abstencao). Ver arca/idioma_deteccao.py."""
    texto = str(texto or "").strip()
    if len(texto) < DETECCAO_MINIMO_CHARS:
        return None
    return _deteccao.detectar(texto)[0]


def detectar_idioma_com_confianca(texto):
    """(lang1, conf1, lang2, conf2) no formato antigo, a partir do detector novo.

    conf vem da margem entre o 1o e o 2o idioma; lang1 e None quando o
    detector se abstem (nesse caso quem chama nao deve alterar nada).
    """
    texto = str(texto or "").strip()
    if len(texto) < DETECCAO_MINIMO_CHARS:
        return None, 0.0, None, 0.0
    pontos, _n = _deteccao.pontuar(texto)
    ordem = sorted(pontos.items(), key=lambda kv: -kv[1])
    lang, margem = _deteccao.detectar(texto)
    conf = min(0.99, 0.5 + margem) if lang else 0.0
    return lang, conf, ordem[1][0], max(0.0, 0.5 - margem / 2)


def decidir_idioma(texto, atual, lang_iso=None, usar_iso=True):
    """Decisao UNICA do idioma de um titulo/descricao (modo CONSERTAR).

    1. o detector (arca/idioma_deteccao.py) decidiu com seguranca -> ele;
    2. o detector se absteve e ja ha idioma -> mantem o atual (normalizado);
    3. o detector se absteve e nao ha idioma -> dc.language.iso, se permitido.

    Substitui a cascata antiga (marcadores > diacriticos > detector > ISO, com
    sobrescritas), que acertava 35% dos casos de discordancia; esta acerta 95%
    (amostra rotulada a mao, ver docs/DECISOES.md).
    """
    lang, margem = _deteccao.detectar(texto)
    if lang:
        return lang, "detector (margem %.2f)" % margem
    if atual:
        return normalizar_language(atual), "detector inconclusivo: mantido o atual"
    if usar_iso and lang_iso:
        return lang_iso, "dc.language.iso (detector inconclusivo)"
    return None, None


def eh_par_ambiguo(lang1, lang2):
    if not lang1 or not lang2:
        return False
    par = frozenset({lang1, lang2})
    for par_candidato in Pares_LINGUA_AMBIGUA:
        if par <= par_candidato:
            return True
    return False


# =============================================================================
# MARCADORES LINGUÍSTICOS E DIACRÍTICOS MELHORADOS
# =============================================================================
MARCADORES_POR_IDIOMA = {
    "es": {
        "la", "el", "y", "con", "los", "las", "del", "al", "una", "unos", "unas",
        "un", "en", "también", "pero", "sino", "hasta", "desde", "sobre", "entre",
        "hacia", "para", "por", "sin", "según", "contra", "ante",
        "bajo", "cabe", "durante", "excepto", "salvo", "mediante",
        "tras", "versus", "vía",
    },
    "pt-BR": {
        "e", "com", "os", "as", "do", "da", "uma", "uns", "umas",
        "em", "na", "no", "também", "mas", "porém", "até", "desde", "sobre", "entre",
        "para", "por", "sem", "conforme", "contra", "ante", "sob",
        "durante", "exceto", "salvo", "mediante", "segundo",
        "através", "acerca", "cerca", "perante", "trás",
        "novo", "nova", "inseto", "estudo", "caso",
    },
    "fr": {
        "et", "avec", "les", "des", "du", "de la", "une", "des", "la",
        "mais", "aussi", "pour", "dans", "sur", "entre", "vers",
        "sans", "selon", "contre", "pendant", "excepté", "sauf",
        "par", "depuis", "près", "chez", "au",
    },
    "it": {
        "e", "con", "i", "le", "delle", "degli", "una", "degli",
        "ma", "anche", "per", "in", "su", "tra", "verso", "senza",
        "secondo", "contro", "durante", "eccetto", "salvo",
        "mediante", "presso", "sopra", "sotto",
    },
    "de": {
        "und", "mit", "die", "der", "den", "des", "ein", "eine",
        "aber", "auch", "für", "in", "auf", "zwischen", "zu",
        "ohne", "laut", "gegen", "während", "außer", "bis", "durch",
        "entlang", "für", "gegenüber", "hinter", "neben", "ohne",
        "seit", "über", "ueber", "von", "vor", "zur",
    },
    "en": {
        "and", "with", "the", "a", "an", "but", "also", "for",
        "in", "on", "between", "from", "over", "toward", "without",
        "according", "against", "during", "except", "until",
        "through", "across", "among", "beyond", "per", "via",
        "although", "because", "since", "therefore",
        "of", "to", "is", "are", "was", "were", "be", "been", "being",
        "has", "had", "have", "do", "does", "did", "not", "this", "that",
        "these", "those", "its", "their", "which", "who", "whom",
    },
}


def detectar_por_marcadores(texto, min_score=3):
    """Idioma pelas palavras EXCLUSIVAS de cada lingua; empate = None.

    Antes contava palavras que existem em varias linguas ("para", "sobre",
    "a", "in") e, no empate, vencia a primeira chave do dicionario (es) —
    medido: titulo portugues sem acento virava espanhol.
    """
    if not texto:
        return None, 0
    palavras = re.findall(r"\w+", _deteccao.limpar(texto).lower())
    scores = {l: sum(1 for p in palavras if p in ws) for l, ws in _deteccao._EXCLUSIVAS.items()}
    ordem = sorted(scores.values(), reverse=True)
    melhor = max(scores, key=scores.get)
    if scores[melhor] >= min_score and (len(ordem) < 2 or ordem[0] > ordem[1]):
        return melhor, scores[melhor]
    return None, 0


DIACRITICOS_PT = set('ãõçÃÕÇ')


DIACRITICOS_ES = set('ñüÑÜ')


SUFIXOS_PT = {'ção', 'são', 'dade', 'agem', 'mento', 'ções', 'sões', 'dades', 'agens', 'mentos'}


SUFIXOS_ES = {'ción', 'sión', 'dad', 'aje', 'miento', 'ciones', 'siones', 'dades', 'ajes', 'mientos'}


TERMOS_LATIM = {
    # --- Gêneros/spp biológicos (aparecem em todas as línguas) ---
    'entamoeba', 'leishmania', 'trypanosoma', 'plasmodium', 'schistosoma',
    'lutzomyia', 'phlebotomus', 'triatoma', 'rhodnius', 'panstrongylus',
    'aedes', 'anopheles', 'culex', 'strigomonas', 'angiostrongylus',
    'escherichia', 'salmonella', 'staphylococcus', 'streptococcus',
    'mycobacterium', 'cryptococcus', 'candida', 'aspergillus',
    'histoplasma', 'sporothrix', 'paracoccidioides', 'pneumocystis',
    'toxoplasma', 'giardia', 'cryptosporidium', 'balantidium',
    'echinococcus', 'taenia', 'trichinella', 'ancylostoma', 'necator',
    'strongyloides', 'ascaris', 'enterobius', 'trichuris', 'wuchereria',
    'brugia', 'onchocerca', 'dirofilaria', 'fasciola', 'clonorchis',
    'opisthorchis', 'paragonimus', 'dicrocoelium', 'babesia', 'theileria',
    'sarcocystis', 'isospora', 'cyclospora', 'blastocystis',
    'microsporidium', 'encephalitozoon', 'nosema', 'strigomonas',
    # --- Nomes de espécie frequentes ---
    'braziliensis', 'brasiliensis', 'brasilensis', 'amazonensis',
    'infantum', 'donovani', 'chagasi', 'guyanensis', 'fasciolata',
    'americana', 'viannia', 'lainsoni', 'shawi', 'naiffi',
    'cruzi', 'brucei', 'congolense', 'evansi', 'duttoni',
    'falciparum', 'vivax', 'ovale', 'malariae', 'knowlesi',
    'gallinaceum', 'berghei', 'yoelii', 'chabaudi',
    'mansoni', 'haematobium', 'japonicum', 'mekongi',
    'longipalpis', 'whitmani', 'intermedia', 'neivai', 'pessoai',
    'migonei', 'fischeri', 'shannoni', 'gomezi', 'cruzii',
    'prolixus', 'dimidiata', 'sordida', 'ecuatorianus',
    'major', 'minor',
    'gallicola', 'phacosemoides', 'sicki',
    # --- Ordem/família taxonômica ---
    'diptera', 'psychodidae', 'culicidae', 'simuliidae',
    'hemiptera', 'hymenoptera', 'coleoptera', 'lepidoptera',
    'noctuidae', 'formicidae', 'triatominae', 'reduviidae',
    'nematoda', 'cestoda', 'trematoda', 'arthropoda',
    'annelida', 'mollusca',
    'homoptera', 'psyllidae', 'ciriacreminae',
    # --- Termos médicos/científicos de origem latina ---
    'in vitro', 'in vivo', 'ex vivo', 'in situ',
    'spp', 'sp', 'nº', 'vol', 'pp', 'fig',
}


def remover_termos_latim(texto):
    palavras = re.findall(r'\b\w+\b', texto.lower())
    filtradas = [p for p in palavras if p not in TERMOS_LATIM and len(p) > 2]
    return ' '.join(filtradas)


def curadoria_autor(valor):
    if not valor:
        return valor, None
    original = valor
    texto = valor.strip()
    texto = re.sub(r'\s{2,}', ' ', texto)
    texto = re.sub(r'\s+,', ',', texto)
    texto = re.sub(r',\s+', ', ', texto)
    texto = re.sub(r'\.{2,}', '.', texto)
    texto = re.sub(r',\s*$', '', texto).strip()
    texto = re.sub(r'^\s*,\s*', '', texto).strip()
    if texto != original:
        return texto, original
    return valor, None


def limpar_espacos(valor):
    if not valor:
        return valor, None
    original = valor
    texto = re.sub(r'\s{2,}', ' ', valor)
    texto = re.sub(r'\s+,', ',', texto)
    texto = re.sub(r',\s+', ', ', texto)
    texto = texto.strip()
    if texto != original:
        return texto, original
    return valor, None


def detectar_por_diacriticos(texto):
    pt_count = sum(1 for c in texto if c in DIACRITICOS_PT)
    es_count = sum(1 for c in texto if c in DIACRITICOS_ES)
    
    # sufixo e o FIM de cada palavra (antes testava o inicio: nunca disparava)
    palavras = re.findall(r"\w+", texto.lower())
    pt_sufixos = sum(1 for p in palavras for suf in SUFIXOS_PT if p.endswith(suf))
    es_sufixos = sum(1 for p in palavras for suf in SUFIXOS_ES if p.endswith(suf))
    
    score_pt = pt_count + pt_sufixos
    score_es = es_count + es_sufixos
    
    if score_pt > score_es and score_pt >= 2:
        return "pt-BR", score_pt
    if score_es > score_pt and score_es >= 2:
        return "es", score_es
    if score_pt == score_es and score_pt >= 2:
        if pt_count > es_count:
            return "pt-BR", score_pt
        elif es_count > pt_count:
            return "es", score_es
    return None, 0


def ler_item(cliente, item_uuid):
    return cliente.get_json(
        f"{cliente.base_url}/core/items/{item_uuid}", required=True
    )


def _ler_metadata_campo(cliente, item_uuid, campo):
    item = ler_item(cliente, item_uuid)
    return (item.get("metadata") or {}).get(campo) or []


def aplicar_operacao_unica(cliente, item_uuid, info_operacao):
    url = f"{cliente.base_url}/core/items/{item_uuid}"
    campo = info_operacao["campo"]
    indice = info_operacao["indice"]
    novo = info_operacao["novo"]

    valores = _ler_metadata_campo(cliente, item_uuid, campo)
    if indice >= len(valores):
        return
    if novo is not None:
        op_replace = {
            "op": "replace",
            "path": f"/metadata/{campo}/{indice}/language",
            "value": novo,
        }
        try:
            cliente.patch_json(url, [op_replace], required=True)
            return
        except Exception:
            pass

        op_add = {
            "op": "add",
            "path": f"/metadata/{campo}/{indice}/language",
            "value": novo,
        }
        try:
            cliente.patch_json(url, [op_add], required=True)
            return
        except Exception:
            pass
    else:
        op_remove = {
            "op": "remove",
            "path": f"/metadata/{campo}/{indice}/language",
        }
        try:
            cliente.patch_json(url, [op_remove], required=True)
            return
        except Exception:
            pass

    raise ErroHTTPDSpace(
        f"Falha ao aplicar operação em {campo}[{indice}]: "
        f"replace/add/remove language todos falharam"
    )


def aplicar_lote_robusto(cliente, item_uuid, informacoes):
    if not informacoes:
        return
    url = f"{cliente.base_url}/core/items/{item_uuid}"
    operacoes = []
    indices_validos = []
    for info in informacoes:
        valores = _ler_metadata_campo(cliente, item_uuid, info["campo"])
        idx = info["indice"]
        if idx >= len(valores):
            continue
        tipo = info.get("tipo", "idioma")
        if tipo == "valor":
            operacoes.append({
                "op": "replace",
                "path": f"/metadata/{info['campo']}/{idx}/value",
                "value": info["novo"],
            })
        elif info["novo"] is not None:
            operacoes.append({
                "op": "replace",
                "path": f"/metadata/{info['campo']}/{idx}/language",
                "value": info["novo"],
            })
        else:
            operacoes.append({
                "op": "remove",
                "path": f"/metadata/{info['campo']}/{idx}/language",
            })
        indices_validos.append(info)
    if not operacoes:
        return
    try:
        cliente.patch_json(url, operacoes, required=True)
        return
    except Exception:
        if len(operacoes) == 1:
            aplicar_operacao_unica(cliente, item_uuid, indices_validos[0])
            return
        meio = len(operacoes) // 2
        aplicar_lote_robusto(cliente, item_uuid, indices_validos[:meio])
        aplicar_lote_robusto(cliente, item_uuid, indices_validos[meio:])


def verificar_alteracoes(cliente, item_uuid, esperados):
    item = ler_item(cliente, item_uuid)
    metadata = item.get("metadata") or {}
    divergencias = []
    for campo, indice, esperado in esperados:
        valores = metadata.get(campo) or []
        if indice < len(valores):
            real = valores[indice].get("language")
            if real == esperado:
                continue
        encontrado = False
        for v in valores:
            if isinstance(v, dict) and v.get("language") == esperado:
                encontrado = True
                break
        if not encontrado:
            divergencias.append(
                f"{campo}[{indice}]: esperado={esperado!r}, não encontrado"
            )
    if divergencias:
        raise Exception(
            "Verificação pós-PATCH falhou: " + " | ".join(divergencias[:20])
        )


def ajustar_item(cliente, item_uuid, modo="limpar", dry_run=False):
    item = ler_item(cliente, item_uuid)
    metadata = item.get("metadata") or {}
    informacoes = []

    CAMPOS_COM_IDIOMA = {CAMPO_TITLE, CAMPO_TITLE_ALT, CAMPO_PROVENANCE} | CAMPOS_PRESERVAR_IDIOMA | CAMPOS_IDIOMA_FIXO_PT | set(CAMPOS_IDIOMA_FIXO.keys())

    # ============================================================
    # NORMALIZAÇÃO PREVENTIVA: pt_BR → pt-BR em TODOS os campos
    # ============================================================
    # "TODOS" menos os de somente-leitura. Esta varredura roda ANTES do
    # guarda de CAMPOS_SOMENTE_LEITURA que existe no laco principal mais
    # abaixo, entao sem este `continue` ela era a unica porta aberta:
    # o fluxo de idiomas (D1-D4) reescrevia o atributo `language` de
    # qualquer valor de dcterms.provenance que viesse com "pt_BR" ou
    # "pt_br" — ou seja, mexia numa linha que nao e' nossa. Nao apagava o
    # valor, mas alterava metadado do DSpace, que e' justamente o que a
    # regra inegociavel proibe.
    for campo, valores in metadata.items():
        if not isinstance(valores, list):
            continue
        if campo in CAMPOS_SOMENTE_LEITURA:
            continue
        for indice, valor in enumerate(valores):
            if not isinstance(valor, dict):
                continue
            lang_atual = valor.get("language")
            if lang_atual:
                lang_normalizado = normalizar_language(lang_atual)
                if lang_normalizado != lang_atual:
                    informacoes.append({
                        "campo": campo,
                        "indice": indice,
                        "anterior": lang_atual,
                        "novo": lang_normalizado,
                        "motivo": f"normalização preventiva BCP 47: {lang_atual!r} → {lang_normalizado!r}",
                        "valor_original": valor,
                    })
                    metadata[campo][indice]['language'] = lang_normalizado

    lang_iso = None
    iso_vals = metadata.get("dc.language.iso") or []
    if iso_vals and isinstance(iso_vals[0], dict):
        iso_code = (iso_vals[0].get("value") or "").strip()
        if iso_code:
            lang_iso = normalizar_language(iso_code)

    traducoes_encontradas = []
    if modo == "correcao":
        titulos_principais = metadata.get(CAMPO_TITLE) or []
        titulos_alternativos = metadata.get(CAMPO_TITLE_ALT) or []

        # Detecta idioma do primeiro título (referência principal)
        lang_title, conf_title, lang_title_2, conf_title_2 = None, 0.0, None, 0.0
        if titulos_principais and isinstance(titulos_principais[0], dict):
            texto_title = titulos_principais[0].get("value", "")
            if texto_title:
                lang_title, conf_title, lang_title_2, conf_title_2 = detectar_idioma_com_confianca(texto_title)

        # ============================================================
        # DETECÇÃO DE TRADUÇÕES COMPARANDO COM TODOS OS dc.title
        # ============================================================
        for idx_alt, alt in enumerate(titulos_alternativos):
            if not isinstance(alt, dict):
                continue
            texto_alt = alt.get("value", "")
            if not texto_alt:
                continue

            lang_alt_raw = alt.get("language")
            lang_alt = normalizar_language(lang_alt_raw) if lang_alt_raw else None

            lang_marc, score_marc = detectar_por_marcadores(texto_alt)
            lang_diac, score_diac = detectar_por_diacriticos(texto_alt)

            # idioma do alternative pelo detector novo; se ele se abstem, fica o gravado
            lang_detectado_alt = detectar_idioma(texto_alt)
            if lang_detectado_alt:
                lang_alt = lang_detectado_alt

            eh_traducao = False
            eh_duplicata = False
            lang_final = lang_alt
            motivo_traducao = ""

            # --- Verifica duplicata com QUALQUER dc.title ---
            for idx_t, t in enumerate(titulos_principais):
                if not isinstance(t, dict):
                    continue
                texto_t = t.get("value", "")
                if not texto_t:
                    continue
                if texto_t.strip().lower() == texto_alt.strip().lower():
                    eh_duplicata = True
                    lang_t = t.get("language")
                    if lang_t and lang_t != lang_alt:
                        informacoes.append({
                            "campo": CAMPO_TITLE_ALT,
                            "indice": idx_alt,
                            "anterior": lang_alt,
                            "novo": lang_t,
                            "motivo": f"duplicata do dc.title[{idx_t}] com idioma corrigido: {lang_alt!r} → {lang_t!r}",
                            "valor_original": alt,
                        })
                    break

            if eh_duplicata:
                continue

            # --- Compara com TODOS os dc.title para detectar tradução ---
            melhor_similaridade = 0.0
            melhor_idx_t = -1
            melhor_lang_t = None

            for idx_t, t in enumerate(titulos_principais):
                if not isinstance(t, dict):
                    continue
                texto_t = t.get("value", "")
                if not texto_t:
                    continue
                sim = SequenceMatcher(None, texto_t.lower(), texto_alt.lower()).ratio()
                if sim > melhor_similaridade:
                    melhor_similaridade = sim
                    melhor_idx_t = idx_t
                    melhor_lang_t = t.get("language")

            # Se a melhor similaridade for com um título em idioma diferente, é tradução
            if melhor_idx_t != -1 and melhor_lang_t and melhor_lang_t != lang_alt:
                # PT ↔ ES: limiar alto (0.95) porque são muito parecidos
                if (melhor_lang_t == 'pt-BR' and lang_alt == 'es') or (melhor_lang_t == 'es' and lang_alt == 'pt-BR'):
                    if melhor_similaridade < 0.95:
                        eh_traducao = True
                        lang_final = lang_alt
                        motivo_traducao = f"tradução {melhor_lang_t}↔{lang_alt} (sim={melhor_similaridade:.2f}) com título {melhor_idx_t}"
                # EN ↔ ES ou EN ↔ PT
                elif melhor_lang_t == 'en' and (lang_alt == 'es' or lang_alt == 'pt-BR'):
                    if melhor_similaridade < 0.85:
                        eh_traducao = True
                        lang_final = lang_alt
                        motivo_traducao = f"tradução EN→{lang_alt} (sim={melhor_similaridade:.2f}) com título {melhor_idx_t}"
                # Outros casos: similaridade < 0.65
                elif melhor_similaridade < 0.65:
                    eh_traducao = True
                    lang_final = lang_alt
                    motivo_traducao = f"tradução {melhor_lang_t}→{lang_alt} (sim={melhor_similaridade:.2f}) com título {melhor_idx_t}"

            # --- Se não detectou, usa marcadores/diacríticos como fallback ---
            if not eh_traducao and lang_title and lang_alt and lang_title != lang_alt:
                if lang_marc and lang_marc != lang_title and score_marc >= 2 and melhor_similaridade < 0.85:
                    eh_traducao = True
                    lang_final = lang_marc
                    motivo_traducao = f"tradução via marcadores: {lang_title}→{lang_marc} ({score_marc} matches, sim={melhor_similaridade:.2f})"
                elif lang_diac and lang_diac != lang_title and score_diac >= 2 and melhor_similaridade < 0.85:
                    eh_traducao = True
                    lang_final = lang_diac
                    motivo_traducao = f"tradução via diacríticos: {lang_title}→{lang_diac} ({score_diac} ocorrências, sim={melhor_similaridade:.2f})"

            # --- Cross-reference com abstracts (mantido) ---
            if not eh_traducao and lang_title and lang_alt and lang_title != lang_alt:
                abstracts = metadata.get("dc.description.abstract") or []
                for ab in abstracts:
                    if not isinstance(ab, dict):
                        continue
                    lang_ab = ab.get("language")
                    texto_ab = (ab.get("value") or "").strip()
                    if not texto_ab:
                        continue
                    if lang_ab and lang_ab != lang_alt:
                        palavras_alt = set(re.findall(r'\w+', texto_alt.lower()))
                        palavras_ab = set(re.findall(r'\w+', texto_ab.lower()))
                        if palavras_alt and palavras_ab and len(palavras_alt & palavras_ab) >= 5:
                            eh_traducao = True
                            lang_final = lang_ab
                            motivo_traducao = f"tradução via abstract: {lang_alt}→{lang_ab}"
                            break
                    if not lang_ab and len(texto_ab) > 100:
                        lang_ab_det, _, _, _ = detectar_idioma_com_confianca(texto_ab)
                        if lang_ab_det and lang_ab_det != lang_alt:
                            palavras_alt = set(re.findall(r'\w+', texto_alt.lower()))
                            palavras_ab = set(re.findall(r'\w+', texto_ab.lower()))
                            if palavras_alt and palavras_ab and len(palavras_alt & palavras_ab) >= 5:
                                eh_traducao = True
                                lang_final = lang_ab_det
                                motivo_traducao = f"tradução via abstract (detectado): {lang_alt}→{lang_ab_det}"
                                break

            if eh_traducao:
                traducoes_encontradas.append({
                    "indice_alt": idx_alt,
                    "texto": texto_alt,
                    "lang_detectado": lang_final,
                    "lang_title": lang_title,
                    "motivo": motivo_traducao,
                    "similaridade": melhor_similaridade,
                })

        # --- Detecção de traduções entre múltiplos dc.title (já existente) ---
        if len(titulos_principais) > 1:
            for idx_t, t in enumerate(titulos_principais):
                if not isinstance(t, dict):
                    continue
                texto_t = t.get("value", "")
                if not texto_t:
                    continue
                lang_t, conf_t, lang_t_2, conf_t_2 = detectar_idioma_com_confianca(texto_t)
                if lang_t is None:
                    continue
                texto_t_norm = texto_t.strip().lower()
                duplicado = any(
                    (t2.get("value") or "").strip().lower() == texto_t_norm
                    for i2, t2 in enumerate(titulos_principais)
                    if i2 < idx_t and isinstance(t2, dict)
                )
                if duplicado:
                    if lang_t and lang_t != (t.get("language") or ""):
                        if conf_t >= LIMIAR_CONFIANCA_ALTA or not eh_par_ambiguo(lang_t, lang_t_2):
                            informacoes.append({
                                "campo": CAMPO_TITLE,
                                "indice": idx_t,
                                "anterior": t.get("language"),
                                "novo": lang_t,
                                "motivo": f"duplicado em dc.title corrigido: {t.get('language')!r} -> {lang_t!r}",
                                "valor_original": t,
                            })
                    continue
                primeiro_texto = titulos_principais[0].get("value", "") if titulos_principais else ""
                primeiro_lang = detectar_idioma(primeiro_texto) if primeiro_texto else None
                if primeiro_lang and lang_t != primeiro_lang:
                    if conf_t < LIMIAR_CONFIANCA_ALTA and eh_par_ambiguo(lang_t, lang_t_2):
                        continue
                    similaridade = SequenceMatcher(None, primeiro_texto.lower(), texto_t.lower()).ratio()
                    if similaridade < 0.75:
                        traducoes_encontradas.append({
                            "indice_alt": idx_t,
                            "texto": texto_t,
                            "lang_detectado": lang_t,
                            "lang_title": primeiro_lang,
                            "campo_origem": CAMPO_TITLE,
                            "motivo": f"tradução entre dc.titles: {primeiro_lang}→{lang_t} (sim={similaridade:.2f})",
                            "similaridade": similaridade,
                        })

    # ============================================================
    # PROCESSAMENTO DOS CAMPOS (LIMPAR e CORREÇÃO)
    # ============================================================
    for campo, valores in metadata.items():
        if not isinstance(valores, list):
            continue
        # Provenance fica fora da trilha de idiomas, nos dois sentidos: nao e
        # corrigida e nao e reescrita. Antes havia aqui uma regra que forçava
        # language="en" em toda provenance — o que, em outro item, significaria
        # reescrever linha do workflow do DSpace. Neste repositorio o DSpace ja
        # devolve "en" sozinho, entao a regra nunca produziu mudanca visivel,
        # mas o risco nao valia a pena.
        if campo in CAMPOS_SOMENTE_LEITURA:
            continue
        for indice, valor in enumerate(valores):
            if not isinstance(valor, dict):
                continue
            atual = valor.get("language")
            novo = atual
            motivo = None

            if modo == "limpar":
                if campo in CAMPOS_IDIOMA_FIXO_PT:
                    if atual != "pt-BR":
                        novo = "pt-BR"
                        motivo = "idioma fixo pt-BR para vocabulário controlado"
                elif campo in CAMPOS_IDIOMA_FIXO:
                    lang_fixo = CAMPOS_IDIOMA_FIXO[campo]
                    if atual != lang_fixo:
                        novo = lang_fixo
                        motivo = f"idioma fixo {lang_fixo} para {campo}"
                elif campo not in CAMPOS_COM_IDIOMA and atual is not None:
                    novo = None
                    motivo = "remoção de idioma em campo não permitido"
                elif campo in CAMPOS_COM_IDIOMA:
                    texto = valor.get("value") or ""
                    if atual:
                        normalizado = normalizar_language(atual)
                        if normalizado != atual:
                            novo = normalizado
                            motivo = f"normalização BCP 47: {atual!r} → {normalizado!r}"
                    elif not atual and texto and len(texto) >= DETECCAO_MINIMO_CHARS:
                        lang_detectado = detectar_idioma(texto)
                        if lang_detectado:
                            novo = lang_detectado
                            motivo = f"idioma preenchido via detecção: {atual!r} → {lang_detectado!r}"

            elif modo == "correcao":
                # ============================================================
                # CAMPO dc.title.alternative (NUNCA usa ISO)
                # ============================================================
                if campo == CAMPO_TITLE_ALT:
                    if any(t["indice_alt"] == indice for t in traducoes_encontradas):
                        continue

                    texto = valor.get("value") or ""
                    if not texto or len(texto) < DETECCAO_MINIMO_CHARS:
                        continue

                    # nunca usa dc.language.iso no alternative
                    ref, fonte = decidir_idioma(texto, atual, usar_iso=False)

                    if ref is not None and ref != atual:
                        novo = ref
                        motivo = f"idioma {'corrigido' if atual else 'preenchido'} via {fonte}: {atual!r} → {ref!r}"
                    elif not atual:
                        pass
                    elif atual:
                        normalizado = normalizar_language(atual)
                        if normalizado != atual:
                            novo = normalizado
                            motivo = f"normalização BCP 47: {atual!r} → {normalizado!r}"

                # ============================================================
                # CAMPO dc.title (marcadores/diacríticos sobrescrevem ISO)
                # ============================================================
                elif campo == CAMPO_TITLE:
                    texto = valor.get("value") or ""
                    if not texto or len(texto) < DETECCAO_MINIMO_CHARS:
                        continue

                    ref, fonte = decidir_idioma(texto, atual, lang_iso)

                    if ref is not None and ref != atual:
                        novo = ref
                        motivo = f"idioma {'corrigido' if atual else 'preenchido'} via {fonte}: {atual!r} → {ref!r}"
                    elif atual:
                        normalizado = normalizar_language(atual)
                        if normalizado != atual:
                            novo = normalizado
                            motivo = f"normalização BCP 47: {atual!r} → {normalizado!r}"

                elif campo in CAMPOS_PRESERVAR_IDIOMA:
                    texto = valor.get("value") or ""
                    ref, fonte = decidir_idioma(texto, atual, lang_iso)

                    if ref is not None and ref != atual:
                        novo = ref
                        motivo = f"idioma {'corrigido' if atual else 'preenchido'} via {fonte}: {atual!r} → {ref!r}"

            if motivo is not None and novo != atual:
                informacoes.append({
                    "campo": campo,
                    "indice": indice,
                    "anterior": atual,
                    "novo": novo,
                    "motivo": motivo,
                    "valor_original": valor,
                })

            if campo == CAMPO_CONTRIBUTOR_AUTHOR:
                texto_atual = valor.get("value") or ""
                texto_limpo, texto_original = curadoria_autor(texto_atual)
                if texto_original is not None:
                    informacoes.append({
                        "campo": campo,
                        "indice": indice,
                        "anterior": texto_original,
                        "novo": texto_limpo,
                        "motivo": "curadoria autor: espaços/pontos extras removidos",
                        "valor_original": valor,
                        "tipo": "valor",
                    })

            if campo not in (CAMPO_CONTRIBUTOR_AUTHOR,) + CAMPOS_SOMENTE_LEITURA:
                texto_atual = valor.get("value") or ""
                texto_limpo, texto_original = limpar_espacos(texto_atual)
                if texto_original is not None:
                    informacoes.append({
                        "campo": campo,
                        "indice": indice,
                        "anterior": texto_original,
                        "novo": texto_limpo,
                        "motivo": "limpeza: espaços duplos removidos",
                        "valor_original": valor,
                        "tipo": "valor",
                    })

    # Aplica PATCHes e registra
    if informacoes:
        if not dry_run:
            for inicio in range(0, len(informacoes), PATCH_BATCH_SIZE):
                aplicar_lote_robusto(
                    cliente,
                    item_uuid,
                    informacoes[inicio:inicio + PATCH_BATCH_SIZE],
                )
            verificar_alteracoes(
                cliente,
                item_uuid,
                [(i["campo"], i["indice"], i["novo"]) for i in informacoes],
            )
        for info in informacoes:
            RELATORIO.registrar(
                item_uuid, info["campo"], info["indice"],
                info["anterior"], info["novo"], info["motivo"],
            )

    movidos, principal = [], None
    if modo == "correcao":
        movidos = mover_traducoes_para_title(cliente, item_uuid, metadata, dry_run=dry_run)
        principal = titulo_principal_no_iso(cliente, item_uuid, metadata, lang_iso, informacoes, movidos,
                                            dry_run=dry_run)

    if not informacoes and not movidos and not principal:
        return "sem_alteracoes"
    return "processado"


def _idioma_titulo(t):
    return normalizar_language(t.get("language")) if t.get("language") else detectar_idioma(t.get("value", ""))


def titulo_principal_no_iso(cliente, item_uuid, metadata, lang_iso, informacoes, movidos, dry_run=False):
    """Regra do responsavel (D28, 2026-10-16): o 1o dc.title e o do idioma de dc.language.iso (idioma do texto
    completo). Se o 1o nao esta nesse idioma e outro dc.title esta, esse vai para a 1a posicao; os demais
    mantem a ordem entre si. Sem titulo no idioma do ISO: nao mexe. Devolve (de, texto) ou None."""
    if not lang_iso:
        return None
    if dry_run:                               # reproduz o que os passos anteriores teriam gravado
        titulos = [dict(t) for t in (metadata.get(CAMPO_TITLE) or []) if isinstance(t, dict)]
        for info in informacoes:
            if info["campo"] == CAMPO_TITLE and info.get("tipo", "idioma") == "idioma" and info["indice"] < len(titulos):
                titulos[info["indice"]]["language"] = info["novo"]
        titulos += [{"value": texto, "language": lang} for _i, texto, lang in movidos]
    else:
        titulos = (ler_item(cliente, item_uuid).get("metadata") or {}).get(CAMPO_TITLE) or []
    if len(titulos) < 2 or _idioma_titulo(titulos[0]) == lang_iso:
        return None
    de = next((j for j, t in enumerate(titulos) if j > 0 and _idioma_titulo(t) == lang_iso), None)
    if de is None:
        return None
    texto = titulos[de].get("value")
    if not dry_run:
        url = f"{cliente.base_url}/core/items/{item_uuid}"
        cliente.patch_json(url, [{"op": "move", "from": f"/metadata/{CAMPO_TITLE}/{de}",
                                  "path": f"/metadata/{CAMPO_TITLE}/0"}], required=True)
        novo = (ler_item(cliente, item_uuid).get("metadata") or {}).get(CAMPO_TITLE) or []
        if not novo or novo[0].get("value") != texto:
            raise ErroHTTPDSpace("Verificação pós-PATCH falhou: o título no idioma %s não ficou em 1º" % lang_iso)
    RELATORIO.registrar(item_uuid, CAMPO_TITLE, de, f"dc.title[{de}] ({lang_iso})", "dc.title[0]",
                        f"título no idioma de dc.language.iso ({lang_iso}) passa a ser o 1º")
    return (de, texto)


def planejar_traducoes(metadata):
    """Titulos de dc.title.alternative que sao traducao e devem ir para dc.title (decisao do responsavel):
    o idioma do alternativo (pelo detector; se ele se abstem, nao mexe) ainda nao existe no dc.title.
    Se ja existe titulo naquele idioma, fica onde esta. Devolve [(indice_alt, texto, idioma)]."""
    titulos = metadata.get(CAMPO_TITLE) or []
    presentes = set()
    for t in titulos:
        if not isinstance(t, dict):
            continue
        lang = normalizar_language(t.get("language")) if t.get("language") else detectar_idioma(t.get("value", ""))
        if lang:
            presentes.add(lang)
    plano = []
    for idx, alt in enumerate(metadata.get(CAMPO_TITLE_ALT) or []):
        if not isinstance(alt, dict):
            continue
        texto = (alt.get("value") or "").strip()
        lang = detectar_idioma(texto)
        if not texto or not lang or lang in presentes:
            continue
        presentes.add(lang)                     # dois alternativos no mesmo idioma: so o primeiro vai
        plano.append((idx, alt.get("value"), lang))
    return plano


def mover_traducoes_para_title(cliente, item_uuid, metadata, dry_run=False):
    """Acrescenta as traducoes no FIM do dc.title (o 1o titulo continua o 1o) e tira do alternativo."""
    if not dry_run:
        metadata = (ler_item(cliente, item_uuid).get("metadata") or {})   # indices atuais, depois dos PATCHes de idioma
    plano = planejar_traducoes(metadata)
    if not plano:
        return []
    if not dry_run:
        url = f"{cliente.base_url}/core/items/{item_uuid}"
        ops = [{"op": "add", "path": f"/metadata/{CAMPO_TITLE}/-", "value": {"value": texto, "language": lang}}
               for _i, texto, lang in plano]
        ops += [{"op": "remove", "path": f"/metadata/{CAMPO_TITLE_ALT}/{i}"}
                for i, _t, _l in sorted(plano, key=lambda x: -x[0])]          # remove do maior indice para o menor
        cliente.patch_json(url, ops, required=True)
        md = ler_item(cliente, item_uuid).get("metadata") or {}
        nos_titulos = {(v.get("value"), v.get("language")) for v in md.get(CAMPO_TITLE) or []}
        nos_alts = {v.get("value") for v in md.get(CAMPO_TITLE_ALT) or []}
        faltou = [t for _i, t, l in plano if (t, l) not in nos_titulos or t in nos_alts]
        if faltou:
            raise ErroHTTPDSpace("Verificação pós-PATCH falhou ao mover tradução: %s" % faltou[:3])
    for i, texto, lang in plano:
        RELATORIO.registrar(item_uuid, CAMPO_TITLE_ALT, i, f"dc.title.alternative ({lang})",
                            f"dc.title (fim, {lang})",
                            f"tradução movida para dc.title: idioma {lang} ainda não existia no dc.title")
    return plano


def processar_item_threadsafe(item_uuid, modo="limpar", dry_run=False):
    ultimo_erro = None
    for tentativa in range(1, ITEM_MAX_TENTATIVAS + 1):
        try:
            cliente = get_thread_client()
            resultado = ajustar_item(cliente, item_uuid, modo, dry_run=dry_run)
            return item_uuid, resultado, None, tentativa
        except Exception as exc:
            ultimo_erro = exc
            if tentativa < ITEM_MAX_TENTATIVAS:
                time.sleep(min(2 ** tentativa, 10))
    status = "permissao" if "ERRO_PERMISSAO" in str(ultimo_erro) else "erro"
    return item_uuid, status, ultimo_erro, ITEM_MAX_TENTATIVAS


def executar_lote(uuids, workers, modo="limpar", dry_run=False):
    uuids = list(dict.fromkeys(normalizar_identificador(u) for u in uuids if u))
    resumo = {
        "processado": 0,
        "sem_alteracoes": 0,
        "item_nao_encontrado": 0,
        "permissao": 0,
        "erro": 0,
    }
    if not uuids:
        aviso("Nenhum item para processar.")
        return resumo

    titulo("PROCESSAMENTO")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progresso:
        tarefa = progresso.add_task(
            "Corrigindo idiomas" if modo == "correcao" else "Ajustando idiomas",
            total=len(uuids),
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, workers)
        ) as executor:
            futuros = {
                executor.submit(processar_item_threadsafe, u, modo, dry_run): u
                for u in uuids
            }
            for futuro in concurrent.futures.as_completed(futuros):
                item_uuid, resultado, exc, tentativa = futuro.result()
                resumo[resultado] = resumo.get(resultado, 0) + 1
                if exc is not None:
                    erro(
                        f"{item_uuid}: falhou após {tentativa} tentativa(s): {exc}"
                    )
                progresso.update(tarefa, advance=1)
    return resumo


# =============================================================================
# INTERFACE
# =============================================================================
def mostrar_resumo(resumo):
    titulo("RESUMO FINAL")
    tabela = Table(box=box.ROUNDED)
    tabela.add_column("Resultado", style="cyan")
    tabela.add_column("Qtd.", justify="right")
    nomes = {
        "processado": "Itens atualizados",
        "sem_alteracoes": "Sem alterações necessárias",
        "item_nao_encontrado": "Item não encontrado",
        "permissao": "Sem permissão",
        "erro": "Erros",
    }
    for chave, nome in nomes.items():
        tabela.add_row(nome, str(resumo.get(chave, 0)))
    console.print(tabela)
    console.print(
        Panel.fit(
            f"[bold]Relatório de alterações[/bold]\n"
            f"{RELATORIO.caminho}\n"
            f"Total de registros modificados: {RELATORIO.total_linhas}",
            border_style="green" if RELATORIO.total_linhas > 0 else "yellow",
        )
    )


def configuracao_lote():
    page_size = min(
        MAX_SAFE_PAGE_SIZE,
        max(1, IntPrompt.ask("Page size", default=DEFAULT_PAGE_SIZE)),
    )
    map_workers = max(
        1,
        IntPrompt.ask(
            "Threads de mapeamento", default=DEFAULT_MAPPING_WORKERS
        ),
    )
    process_workers = max(
        1,
        IntPrompt.ask(
            "Threads de processamento", default=DEFAULT_PROCESS_WORKERS
        ),
    )
    return page_size, map_workers, process_workers


def configurar_data():
    console.print(
        "[bold cyan]Filtro de data (opcional)[/bold cyan]\n"
        "  Processa apenas itens com [bold]lastModified[/bold] ou "
        "[bold]dc.date.accessioned[/bold] a partir da data informada.\n"
        "  Formato: DD/MM/AAAA  |  Enter vazio = sem filtro."
    )
    texto = Prompt.ask("Data mínima", default="").strip()
    if not texto:
        return None
    formatos = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y"]
    for fmt in formatos:
        try:
            return datetime.strptime(texto, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Data '{texto}' inválida. Use DD/MM/AAAA (ex: 01/01/2025)."
    )


def _parse_date_iso(texto):
    if not texto:
        return None
    texto = str(texto).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(texto, fmt)
            return dt.date() if isinstance(dt, datetime) else dt
        except ValueError:
            continue
    if len(texto) >= 10:
        try:
            return datetime.strptime(texto[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


def _item_passa_filtro_data(item, data_inicio):
    if data_inicio is None:
        return True
    last_mod = _parse_date_iso(item.get("lastModified"))
    if last_mod and last_mod >= data_inicio:
        return True
    accessioned_vals = (item.get("metadata") or {}).get("dc.date.accessioned") or []
    for v in accessioned_vals:
        if isinstance(v, dict):
            d = _parse_date_iso(v.get("value"))
            if d and d >= data_inicio:
                return True
    return False


def filtrar_por_data(itens, data_inicio):
    if data_inicio is None:
        return itens, len(itens), 0
    filtrados = [i for i in itens if _item_passa_filtro_data(i, data_inicio)]
    removidos = len(itens) - len(filtrados)
    return filtrados, len(filtrados), removidos


def confirmar(descricao, quantidade=None):
    tabela = Table(title="Conferência", box=box.ROUNDED)
    tabela.add_column("Campo", style="cyan")
    tabela.add_column("Valor")
    tabela.add_row("Operação", descricao)
    if quantidade is not None:
        tabela.add_row("Itens", str(quantidade))
    tabela.add_row(
        "Preserva/normaliza/detecta",
        " | ".join(sorted(CAMPOS_PRESERVAR_IDIOMA)),
    )
    tabela.add_row(CAMPO_PROVENANCE,
                   "registro interno do script (proibido: %s)"
                   % CAMPO_PROVENANCE_DSPACE)
    tabela.add_row("dc.description.abstract*", "idioma fixo (pt-BR/en/fr/es)")
    tabela.add_row("dc.subject.*", "idioma fixo (pt-BR/en/fr/es/mesh→en)")
    tabela.add_row(CAMPO_CONTRIBUTOR_AUTHOR, "curadoria: remove espaços/pontos extras")
    tabela.add_row("Todos os outros metadados", "sempre language=null")
    console.print(tabela)
    aviso("Esta operação alterará o repositório diretamente.")
    return Confirm.ask("Iniciar?", default=True)


def ler_lista_digitada():
    console.print(
        "Cole UUIDs, handles ou URLs. Separe por espaço, vírgula ou linha. "
        "Linha vazia finaliza."
    )
    linhas = []
    while True:
        linha = input("> ").strip()
        if not linha:
            break
        linhas.append(linha)
    return [p for p in re.split(r"[\s,;]+", "\n".join(linhas)) if p]


def ler_arquivo_uuids(caminho):
    with open(caminho, "r", encoding="utf-8-sig") as arq:
        encontrados = re.findall(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            arq.read(),
        )
    return list(dict.fromkeys(encontrados))


def fluxo_todos():
    page_size, map_workers, process_workers = configuracao_lote()
    data_inicio = configurar_data()
    if not confirmar("Todos os itens"):
        return
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    uuids = mapear_todos(cliente, page_size, map_workers, data_inicio=data_inicio)
    mostrar_resumo(executar_lote(uuids, process_workers))


def fluxo_colecao():
    page_size, map_workers, process_workers = configuracao_lote()
    data_inicio = configurar_data()
    entrada = Prompt.ask("UUID, handle ou URL da coleção")
    if not confirmar(f"Coleção {entrada}"):
        return
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    uuids = mapear_colecao(cliente, entrada, page_size, map_workers, data_inicio=data_inicio)
    mostrar_resumo(executar_lote(uuids, process_workers))


def fluxo_um_item():
    entrada = Prompt.ask("UUID, handle ou URL do item")
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    item = resolver_item(cliente, entrada)
    if not item or not item.get("uuid"):
        erro("Item não encontrado.")
        return
    if not confirmar("Um item", 1):
        return
    mostrar_resumo(executar_lote([item["uuid"]], 1))


def fluxo_lista():
    workers = max(
        1,
        IntPrompt.ask(
            "Threads de processamento", default=DEFAULT_PROCESS_WORKERS
        ),
    )
    entradas = ler_lista_digitada()
    if not confirmar("Lista digitada", len(entradas)):
        return
    cookies = obter_cookies_cdp()
    mostrar_resumo(executar_lote(entradas, workers))


def fluxo_arquivo_erros():
    workers = max(
        1,
        IntPrompt.ask(
            "Threads de processamento", default=DEFAULT_PROCESS_WORKERS
        ),
    )
    caminho = Prompt.ask("Caminho completo do TXT de UUIDs com erro").strip('"')
    uuids = ler_arquivo_uuids(caminho)
    if not confirmar("Reprocessar arquivo de erros", len(uuids)):
        return
    cookies = obter_cookies_cdp()
    mostrar_resumo(executar_lote(uuids, workers))


def confirmar_correcao(descricao, quantidade=None):
    tabela = Table(title="Conferência — Correção de Idiomas", box=box.ROUNDED)
    tabela.add_column("Campo", style="cyan")
    tabela.add_column("Valor")
    tabela.add_row("Operação", descricao)
    tabela.add_row("Modo", "CORREÇÃO (detecta e corrige idiomas incorretos)")
    if quantidade is not None:
        tabela.add_row("Itens", str(quantidade))
    tabela.add_row(
        "Campos detectados",
        " | ".join(sorted(CAMPOS_PRESERVAR_IDIOMA)),
    )
    tabela.add_row(CAMPO_PROVENANCE,
                   "registro interno do script (proibido: %s)"
                   % CAMPO_PROVENANCE_DSPACE)
    tabela.add_row("dc.description.abstract*", "idioma fixo (pt-BR/en/fr/es)")
    tabela.add_row("dc.subject.*", "idioma fixo (pt-BR/en/fr/es/mesh→en)")
    tabela.add_row(CAMPO_CONTRIBUTOR_AUTHOR, "curadoria: remove espaços/pontos extras")
    tabela.add_row("Todos os outros metadados", "sempre language=null")
    console.print(tabela)
    aviso("Esta operação corrigirá idiomas incorretos no repositório.")
    return Confirm.ask("Iniciar?", default=True)


def fluxo_consertar_todos():
    page_size, map_workers, process_workers = configuracao_lote()
    data_inicio = configurar_data()
    if not confirmar_correcao("Consertar idiomas — Todos os itens"):
        return
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    uuids = mapear_todos(cliente, page_size, map_workers, data_inicio=data_inicio)
    mostrar_resumo(executar_lote(uuids, process_workers, modo="correcao"))


def fluxo_consertar_colecao():
    page_size, map_workers, process_workers = configuracao_lote()
    data_inicio = configurar_data()
    entrada = Prompt.ask("UUID, handle ou URL da coleção")
    if not confirmar_correcao(f"Consertar idiomas — Coleção {entrada}"):
        return
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    uuids = mapear_colecao(cliente, entrada, page_size, map_workers, data_inicio=data_inicio)
    mostrar_resumo(executar_lote(uuids, process_workers, modo="correcao"))


def fluxo_consertar_um_item():
    entrada = Prompt.ask("UUID, handle ou URL do item")
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    item = resolver_item(cliente, entrada)
    if not item or not item.get("uuid"):
        erro("Item não encontrado.")
        return
    if not confirmar_correcao("Consertar idiomas — Um item", 1):
        return
    mostrar_resumo(executar_lote([item["uuid"]], 1, modo="correcao"))


def fluxo_consertar_lista():
    workers = max(
        1,
        IntPrompt.ask(
            "Threads de processamento", default=DEFAULT_PROCESS_WORKERS
        ),
    )
    entradas = ler_lista_digitada()
    if not confirmar_correcao("Consertar idiomas — Lista digitada", len(entradas)):
        return
    cookies = obter_cookies_cdp()
    mostrar_resumo(executar_lote(entradas, workers, modo="correcao"))


def fluxo_simular_todos():
    page_size, map_workers, process_workers = configuracao_lote()
    data_inicio = configurar_data()
    console.print("[bold yellow]SIMULAÇÃO — Nenhuma alteração será aplicada[/bold yellow]")
    if not Confirm.ask("Simular correção em todos os itens?", default=True):
        return
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    uuids = mapear_todos(cliente, page_size, map_workers, data_inicio=data_inicio)
    mostrar_resumo(executar_lote(uuids, process_workers, modo="correcao", dry_run=True))


def fluxo_simular_colecao():
    page_size, map_workers, process_workers = configuracao_lote()
    data_inicio = configurar_data()
    entrada = Prompt.ask("UUID, handle ou URL da coleção")
    console.print("[bold yellow]SIMULAÇÃO — Nenhuma alteração será aplicada[/bold yellow]")
    if not Confirm.ask(f"Simular correção na coleção {entrada}?", default=True):
        return
    cookies = obter_cookies_cdp()
    cliente = criar_cliente(cdp_cookies=cookies)
    cliente.login()
    uuids = mapear_colecao(cliente, entrada, page_size, map_workers, data_inicio=data_inicio)
    mostrar_resumo(executar_lote(uuids, process_workers, modo="correcao", dry_run=True))
