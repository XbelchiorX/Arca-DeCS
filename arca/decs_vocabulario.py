# -*- coding: utf-8 -*-
"""Vocabulario DeCS: as duas fontes de verdade e como elas se encontram.

Ha DUAS coisas diferentes chamadas "vocabulario", e confundir as duas foi a
origem de varios bugs antigos:

1. A BASE DeCS (BIREME) — o que cada descritor SIGNIFICA.
   Fonte: o XML oficial da BIREME (formato NLM DescriptorRecordSet),
   ex.: bireme_decs_por2026.xml. Dele saem nome PT/EN, sinonimos PT/EN/ES,
   nota de escopo, anotacao para indexador, numeros de arvore, qualificadores
   permitidos e combinacoes de entrada (descritor/qualificador -> descritor).

2. A ARVORE DO SERVIDOR (DSpace) — o que pode ser GRAVADO.
   Fonte: o vocabulario `decs` que o DSpace do ARCA serve, baixado por
   ferramentas/decs/baixar_vocabulario_servidor.py. Cada no tem id = numero de
   arvore (ex.: "M01.975.807") e valor = caminho sem a categoria de topo
   (ex.: "Pessoas::Mulheres::Gestantes"). Um valor de dc.subject.decs so e
   reconhecido pelo DSpace se for IDENTICO a um desses valores.

A ponte entre os dois e o NUMERO DE ARVORE: o descritor D037841 (Gestantes)
tem as arvores M01.975.807 e SP3.522.561.200.488 na base; cada uma delas e
um no do servidor, com o seu caminho gravavel.
"""
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict

from . import config


# ---------------------------------------------------------------------------
# Normalizacao de texto para comparacao (nunca para gravar)
# ---------------------------------------------------------------------------
def norm(texto):
    """Chave de comparacao: NFC, sem acento, minusculas, espacos simples."""
    t = unicodedata.normalize("NFKD", texto or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\s+", " ", t).strip().casefold()
    return t


_PALAVRAS_VAZIAS = {"de", "da", "do", "das", "dos", "e", "a", "o", "as", "os",
                    "em", "na", "no", "of", "the", "and", "y", "la", "el", "en"}


def chave_variante(texto):
    """Conjunto ordenado de palavras: 'Crianca, pre-escolar' == 'Pre-Escolar Crianca'."""
    palavras = re.findall(r"\w+", norm(texto))
    return " ".join(sorted(p for p in palavras if p not in _PALAVRAS_VAZIAS))


_RE_CATEGORIA = re.compile(r"^[^:]*\[[A-Z]{1,2}\]$")


_NOMES_CATEGORIA = {norm(n) for n in config.DECS_CATEGORIAS.values()}


def chave_sem_categoria(valor):
    """Caminho normalizado SEM o primeiro nivel se ele for uma categoria de topo.

    'DOENCAS [C]::Infeccoes::Zoonoses' e 'Infeccoes::Zoonoses' dao a mesma chave.
    """
    niveis = [n.strip() for n in (valor or "").split("::")]
    if len(niveis) > 1 and (_RE_CATEGORIA.match(niveis[0])
                            or norm(niveis[0]) in _NOMES_CATEGORIA):
        niveis = niveis[1:]
    return norm("::".join(niveis))


def _nfc(texto):
    return unicodedata.normalize("NFC", (texto or "").strip())


def categoria_de_topo(arvore):
    """'SP3.522.561' -> 'SP'; 'M01.975' -> 'M'."""
    return re.match(r"[A-Z]+", arvore).group(0) if arvore else ""


def prefixos_da_arvore(arvore):
    """'M01.975.807' -> ['M01', 'M01.975', 'M01.975.807']."""
    partes = arvore.split(".")
    return [".".join(partes[:i]) for i in range(1, len(partes) + 1)]


# ---------------------------------------------------------------------------
# 1. Base DeCS a partir do XML da BIREME
# ---------------------------------------------------------------------------
def _carimbo(caminho):
    st = os.stat(caminho)
    return {"arquivo": os.path.basename(caminho), "tamanho": st.st_size,
            "mtime": int(st.st_mtime)}


def _texto(el, caminho):
    x = el.find(caminho)
    return _nfc(x.text) if x is not None and x.text else ""


def _separar_nome(nome):
    """'Gravidez[Pregnancy]' -> ('Gravidez', 'Pregnancy')."""
    m = re.match(r"^(.*)\[(.*)\]$", nome or "")
    if m:
        return _nfc(m.group(1)), _nfc(m.group(2))
    return _nfc(nome), ""


def _idioma_do_termo(term_ui):
    """Prefixo do TermUI da BIREME -> idioma do termo."""
    if term_ui.startswith(("por",)):
        return "pt"
    if term_ui.startswith(("spa",)):
        return "es"
    if term_ui.startswith(("eng", "T")):
        return "en"
    return ""


def _ler_registros_bireme(caminho):
    """Itera (DescriptorRecord) do XML NLM sem carregar 380 MB na memoria."""
    for _ev, el in ET.iterparse(caminho, events=("end",)):
        if el.tag == "DescriptorRecord":
            yield el
            el.clear()


def construir_base_bireme(caminho_por, caminho_spa=None):
    """Le o XML portugues (e opcionalmente o espanhol) e devolve a base.

    Formato de saida (um dict por descritor, chave = DescriptorUI):
        nome      nome preferido em portugues
        en        nome preferido em ingles
        arvores   numeros de arvore (cada um = uma posicao na arvore)
        sin_pt / sin_en / sin_es   termos de entrada (sinonimos)
        escopo    nota de escopo em portugues
        anotacao  instrucao ao indexador ("INDEXADOR: Nao use", etc.)
        quals     {nome do qualificador normalizado: QualifierUI}
        combina   {QualifierUI: DescriptorUI}  (EntryCombination oficial:
                   "Gravidez/complicacoes" deve ser indexado como
                   "Complicacoes na Gravidez")
    """
    base = {}
    for el in _ler_registros_bireme(caminho_por):
        ui = _texto(el, "DescriptorUI")
        nome_pt, nome_en = _separar_nome(_texto(el, "DescriptorName/String"))
        rec = {
            "nome": nome_pt,
            "en": nome_en,
            "arvores": [_nfc(t.text) for t in el.findall("TreeNumberList/TreeNumber") if t.text],
            "sin_pt": [], "sin_en": [], "sin_es": [],
            "escopo": "",
            "anotacao": _texto(el, "Annotation"),
            "quals": {},
            "combina": {},
        }
        for aq in el.findall("AllowableQualifiersList/AllowableQualifier"):
            q_ui = _texto(aq, "QualifierReferredTo/QualifierUI")
            q_nome = _texto(aq, "QualifierReferredTo/QualifierName/String")
            if q_ui and q_nome:
                rec["quals"][norm(q_nome)] = q_ui
        for ec in el.findall("EntryCombinationList/EntryCombination"):
            q_ui = _texto(ec, "ECIN/QualifierReferredTo/QualifierUI")
            alvo = _texto(ec, "ECOUT/DescriptorReferredTo/DescriptorUI")
            if q_ui and alvo:
                rec["combina"][q_ui] = alvo
        vistos = set()
        for conc in el.findall("ConceptList/Concept"):
            if conc.get("PreferredConceptYN") == "Y" and not rec["escopo"]:
                rec["escopo"] = _texto(conc, "ScopeNote")
            for term in conc.findall("TermList/Term"):
                s = _texto(term, "String")
                lang = _idioma_do_termo(_texto(term, "TermUI"))
                if not s or not lang or (lang, s) in vistos:
                    continue
                vistos.add((lang, s))
                if lang == "pt" and s == nome_pt:
                    continue
                if lang == "en" and s == nome_en:
                    continue
                rec["sin_" + lang].append(s)
        base[ui] = rec

    if caminho_spa and os.path.exists(caminho_spa):
        for el in _ler_registros_bireme(caminho_spa):
            ui = _texto(el, "DescriptorUI")
            if ui not in base:
                continue
            nome_es, _ = _separar_nome(_texto(el, "DescriptorName/String"))
            sin = base[ui]["sin_es"]
            if nome_es and nome_es not in sin:
                sin.append(nome_es)
            for term in el.iter("Term"):
                s = _texto(term, "String")
                if s and _idioma_do_termo(_texto(term, "TermUI")) == "es" and s not in sin:
                    sin.append(s)
    return base


def carregar_base(forcar=False):
    """Base DeCS em cache (data/decs_base.json), refeita se o XML mudar."""
    fonte_por = config.BIREME_XML_POR
    fonte_spa = config.BIREME_XML_SPA
    if not os.path.exists(fonte_por):
        raise FileNotFoundError(
            "XML oficial da BIREME nao encontrado em %s. Ajuste BIREME_XML_POR "
            "em arca/config.py (ou a variavel de ambiente DECS_BIREME_POR)." % fonte_por)
    carimbo = {"por": _carimbo(fonte_por),
               "spa": _carimbo(fonte_spa) if os.path.exists(fonte_spa) else None}
    if not forcar and os.path.exists(config.DECS_BASE_JSON):
        with open(config.DECS_BASE_JSON, encoding="utf-8") as f:
            dados = json.load(f)
        if dados.get("carimbo") == carimbo:
            return dados["descritores"]
    print("Construindo a base DeCS a partir do XML da BIREME (1-3 min, uma vez)...", flush=True)
    base = construir_base_bireme(fonte_por, fonte_spa)
    with open(config.DECS_BASE_JSON, "w", encoding="utf-8") as f:
        json.dump({"carimbo": carimbo, "descritores": base}, f, ensure_ascii=False)
    return base


def arvore_da_base(base, com_categoria=False):
    """{numero_de_arvore: caminho} montada so com a BIREME, no formato do servidor.

    O caminho de 'M01.975.807' e o nome do descritor de cada prefixo
    ('M01', 'M01.975', 'M01.975.807') unidos por '::' — exatamente como o
    DSpace monta o valor de um vocabulario hierarquico, sem a categoria de
    topo (que nao e descritor). Serve para gerar o XML do DSpace e para
    comparar a versao publicada no servidor com a da BIREME.
    """
    # Mesma regra do script_conversao_decs_v2.py: se a BIREME da o MESMO
    # numero de arvore a dois descritores (acontece: B03.300.390.400.001), o
    # primeiro do arquivo ocupa o no e os demais viram nos irmaos com id
    # "<numero>-<DescriptorUI>".
    nome_do_tn = {}
    irmaos = []
    for ui, rec in base.items():
        for tn in rec["arvores"]:
            if tn in nome_do_tn:
                irmaos.append((tn, ui, rec["nome"]))
            else:
                nome_do_tn[tn] = rec["nome"]
    arvore = {}
    for tn in nome_do_tn:
        # nivel intermediario sem descritor na BIREME (ex.: SP2.070.315.420.435)
        # e pulado — mesma regra do v2
        nomes = [nome_do_tn[p] for p in prefixos_da_arvore(tn) if p in nome_do_tn]
        if nomes:
            if com_categoria:
                nomes = [config.rotulo_categoria(categoria_de_topo(tn))] + nomes
            arvore[tn] = "::".join(nomes)
    for tn, ui, nome in irmaos:
        pai = arvore.get(tn, "").rsplit("::", 1)[0] if "::" in arvore.get(tn, "") else ""
        arvore["%s-%s" % (tn, ui)] = (pai + "::" if pai else "") + nome
    if com_categoria:
        for cod in {categoria_de_topo(tn) for tn in nome_do_tn}:
            arvore[cod] = config.rotulo_categoria(cod)
    return arvore


# ---------------------------------------------------------------------------
# 2. Arvore do servidor
# ---------------------------------------------------------------------------
def carregar_arvore_servidor(com_ui=False):
    """{numero_de_arvore: valor_gravavel} (e {tn: ui} se com_ui). Baixe com a D6."""
    if not os.path.exists(config.VOCAB_SERVIDOR_JSON):
        raise FileNotFoundError(
            "Copia do vocabulario do servidor nao encontrada (%s). Rode a opcao D6 ou:\n"
            "    python ferramentas/decs/baixar_vocabulario_servidor.py"
            % config.VOCAB_SERVIDOR_JSON)
    with open(config.VOCAB_SERVIDOR_JSON, encoding="utf-8") as f:
        nos = json.load(f)["nos"]
    arvore = {tn: _nfc(no["valor"]) for tn, no in nos.items()}
    if com_ui:
        return arvore, {tn: no.get("ui") for tn, no in nos.items() if no.get("ui")}
    return arvore


# ---------------------------------------------------------------------------
# 3. O vocabulario unificado usado pelo pipeline
# ---------------------------------------------------------------------------
class Vocabulario:
    """Junta base BIREME + arvore do servidor e responde as perguntas do pipeline.

    Perguntas que ele responde:
      - este valor do acervo e um caminho gravavel?            valor_valido()
      - de que descritor e este caminho / termo?               ui_do_caminho(), uis_do_termo()
      - quais caminhos gravaveis este descritor tem?           caminhos()
      - este descritor e polihierarquico entre categorias?     categorias()
    """

    def __init__(self, base, arvore_servidor, ui_servidor=None):
        self.base = base
        self.servidor = arvore_servidor                     # tn -> valor
        self._valor_para_tn = {}                            # norm(valor) -> tn
        self._sem_categoria_para_tn = {}                    # mesmo caminho sem o 1o nivel de categoria
        for tn, valor in arvore_servidor.items():
            self._valor_para_tn.setdefault(norm(valor), tn)
            self._sem_categoria_para_tn.setdefault(chave_sem_categoria(valor), tn)
        # O servidor tem as 21 categorias no topo? (decidido pelo XML publicado)
        self.servidor_com_categoria = any(re.fullmatch(r"[A-Z]{1,2}", tn) for tn in arvore_servidor)

        # tn -> ui. Pela BIREME e, quando o servidor informa o ID na nota do no,
        # pelo servidor (medido em 2026-10-02: 99,8% iguais; nos 10 casos em
        # que a numeracao mudou entre versoes, vale o que o servidor diz do SEU no).
        self._tn_para_ui = {}
        for ui, rec in base.items():
            for tn in rec["arvores"]:
                self._tn_para_ui.setdefault(tn, ui)       # o 1o do arquivo ocupa o no
                if self._tn_para_ui[tn] != ui:
                    self._tn_para_ui["%s-%s" % (tn, ui)] = ui
        for tn, ui in (ui_servidor or {}).items():
            if ui in base and tn in arvore_servidor:
                self._tn_para_ui[tn] = ui
        # ui -> numeros de arvore do servidor, na ordem oficial da BIREME
        self._tns_do_ui = defaultdict(list)
        for ui, rec in base.items():
            for tn in rec["arvores"]:
                for t in (tn, "%s-%s" % (tn, ui)):
                    if t in arvore_servidor and self._tn_para_ui.get(t) == ui:
                        self._tns_do_ui[ui].append(t)
        for tn, ui in self._tn_para_ui.items():
            if tn in arvore_servidor and tn not in self._tns_do_ui[ui]:
                self._tns_do_ui[ui].append(tn)

        # Indice lexico em DOIS niveis, de proposito:
        #   'nome'     nome preferido (PT, EN ou ES)  -> pode ser aplicado
        #   'sinonimo' termo de entrada               -> so vira candidato
        # "Tabaco" e sinonimo de Nicotiana (a planta); aplicar sinonimo direto
        # foi o que pos uma mariposa num artigo sobre cigarro eletronico.
        self.nomes = defaultdict(set)
        self.sinonimos = defaultdict(set)
        # Descritores que existem no DeCS mas nao tem lugar na arvore do
        # servidor: check tags (Masculino, Feminino: sem numero de arvore) e
        # descritores novos ainda nao publicados no DSpace.
        self.sem_arvore = {}
        # Terceiro nivel: VARIANTE, a mesma palavra em outra ordem ou pontuacao.
        # O acervo tem formas invertidas antigas do MeSH ("Crianca, pre-escolar",
        # "Idoso, 80 anos ou mais") que nao batem com nenhum termo literal.
        self.variantes = defaultdict(set)
        for ui, rec in base.items():
            todos = [rec["nome"], rec["en"]] + rec["sin_pt"] + rec["sin_en"] + rec["sin_es"]
            if not self.caminhos(ui):
                for s in todos:
                    if s:
                        self.sem_arvore.setdefault(norm(s), ui)
                continue
            for s in (rec["nome"], rec["en"]):
                if s:
                    self.nomes[norm(s)].add(ui)
            for s in rec["sin_pt"] + rec["sin_en"] + rec["sin_es"]:
                self.sinonimos[norm(s)].add(ui)
            for s in todos:
                if s:
                    self.variantes[chave_variante(s)].add(ui)
        # o nome espanhol preferido esta no inicio de sin_es (ver construir_base)
        # o acervo escreve "prevencao e controle" e "prevencao & controle"
        self.qualificadores = {}
        for rec in base.values():
            for qn, qui in rec["quals"].items():
                self.qualificadores[qn] = qui
                self.qualificadores[qn.replace("&", "e")] = qui

    # -- caminhos -----------------------------------------------------------
    def caminhos(self, ui):
        """Caminhos GRAVAVEIS do descritor, na ordem das arvores da BIREME.

        So entram arvores que existem no servidor. Uma arvore nova do DeCS
        2026 que ainda nao foi publicada no DSpace nao pode ser gravada.
        """
        saida = []
        for tn in self._tns_do_ui.get(ui, ()):
            v = self.servidor.get(tn)
            if v and v not in saida:
                saida.append(v)
        return saida

    def arvore_do_caminho(self, valor):
        return self._valor_para_tn.get(norm(valor))

    def arvore_do_caminho_em_outro_formato(self, valor):
        """O mesmo no, escrito com/sem a categoria de topo ("DOENCAS [C]::")."""
        return self._sem_categoria_para_tn.get(chave_sem_categoria(valor))

    def valor_valido(self, valor):
        return norm(valor) in self._valor_para_tn

    def valor_canonico(self, valor):
        """O valor exatamente como o servidor o escreve (acento/caixa)."""
        tn = self.arvore_do_caminho(valor)
        return self.servidor.get(tn) if tn else None

    def ui_do_caminho(self, valor):
        tn = self.arvore_do_caminho(valor)
        return self._tn_para_ui.get(tn) if tn else None

    def categorias(self, ui):
        """Categorias de topo (A, C, SP...) dos caminhos gravaveis do descritor."""
        return sorted({categoria_de_topo(tn) for tn in self._tns_do_ui.get(ui, ())})

    def tn_de(self, ui, valor):
        tn = self.arvore_do_caminho(valor)
        return tn if tn and self._tn_para_ui.get(tn) == ui else None

    # -- termos ---------------------------------------------------------------
    def uis_do_termo(self, termo):
        """(uis, nivel): nivel 'nome', 'sinonimo', 'variante' ou None."""
        k = norm(termo)
        if k in self.nomes:
            return sorted(self.nomes[k]), "nome"
        if k in self.sinonimos:
            return sorted(self.sinonimos[k]), "sinonimo"
        kv = chave_variante(termo)
        if kv and kv in self.variantes:
            return sorted(self.variantes[kv]), "variante"
        return [], None

    def nome(self, ui):
        return (self.base.get(ui) or {}).get("nome", ui)

    def texto_descritor(self, ui):
        """Texto que representa o descritor para o modelo de embedding."""
        rec = self.base.get(ui) or {}
        partes = [rec.get("nome", ""), rec.get("escopo", "")] + rec.get("sin_pt", [])[:8]
        return " | ".join(p for p in partes if p)


def carregar_vocabulario(permitir_arvore_bireme=False):
    """Vocabulario de trabalho.

    `permitir_arvore_bireme`: sem a copia do servidor, usa a arvore montada da
    BIREME no formato atual do servidor. Serve para AVALIAR (qual descritor),
    nunca para GRAVAR (qual valor exato existe no DSpace).
    """
    base = carregar_base()
    if config.DECS_ALVO_GRAVACAO == "xml_v2":
        # a mesma arvore do decs.xml v2.1 (a D7 confere que sao identicas)
        return Vocabulario(base, arvore_da_base(base, com_categoria=config.DECS_INCLUIR_CATEGORIA))
    if permitir_arvore_bireme and not os.path.exists(config.VOCAB_SERVIDOR_JSON):
        print("AVISO: usando a arvore da BIREME no lugar da do servidor (so para avaliacao).")
        return Vocabulario(base, arvore_da_base(base))
    arvore, uis = carregar_arvore_servidor(com_ui=True)
    return Vocabulario(base, arvore, uis)
