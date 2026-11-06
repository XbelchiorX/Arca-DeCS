# -*- coding: utf-8 -*-
"""Palavra-chave do autor -> descritor DeCS, quando nao bate literalmente (pedido do responsavel, 2026-10-26).

Mesma logica da clusterizacao das palavras-chave (ferramentas/palavras_chave/pc_clusterizar.py, DECISOES §21),
mas aqui o destino e o vocabulario controlado, entao vale mais: plural/singular e ordem das palavras tambem casam
("Leishmanioses" -> Leishmaniose; "Zika virus infeccao" -> Infeccao por Zika virus).

Niveis, do mais seguro ao menos (para no primeiro que casar):
  grafia     caixa, acento, hifen, pontuacao, "&"                      "Covid 19" = "COVID-19"
  ligacao    + artigo/preposicao neutros (nunca "com/sem", nunca conjuncao, nunca a ultima palavra)
  flexao     + singular/plural de cada palavra                          "Infeccoes por Zika Virus"
  ordem      + qualquer ordem (a conjuncao fica na chave: "Educacao e Saude" nao vira "Educacao em Saude")
  digitacao  + uma palavra (7+ letras, mesmo comeco) que nao existe no DeCS com 1 letra a mais/menos/trocada

Chave que aponta para mais de um descritor nao casa (ambigua). So serve como SINAL do classificador, nunca grava
nada sozinho: lex_kw_aprox (grafia, ligacao, ordem) e lex_kw_aprox_fraco (flexao, digitacao), separados porque o
catalogador concorda bem menos com os fracos ("medicines" -> Medicina, "Amazonas" -> Amazona; medido em 26/10).
"""
import os
import re
import unicodedata
from collections import defaultdict

# artigos e preposicoes neutros (os mesmos da clusterizacao); "com", "sem", "contra" mudam o sentido e ficam
LIG = set("""a o as os um uma uns umas de da do das dos em no na nos nas num numa por pelo pela pelos pelas para ao aos
of the a an in for to on at
el la los las un una del al en de por para
le la les l d un une du des de dans au aux pour en""".split())
COORD = {"and", "e", "y", "et", "or", "ou", "o", "u"}
NIVEIS = ("grafia", "ligacao", "flexao", "ordem", "digitacao")
FRACOS = {"flexao", "digitacao"}
_TAXON = re.compile(r"(?:idae|inae|ina|ini|oidea|aceae|ales|ida|ineae|oideae)$")


def _sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if unicodedata.category(c) != "Mn")


def grafia(v):
    """Sem acento, sem caixa, pontuacao/hifen viram espaco."""
    s = _sem_acento(v).casefold().replace("&", " and ")
    return " ".join(re.sub(r"[\W_]+", " ", s).split())


def conteudo(v):
    """Palavras de conteudo na ordem. Sai so palavra de ligacao solta, em minuscula e que nao e a ultima
    ("Hepatite A", "Vitamina A" e "a-Amilase" ficam)."""
    pal = _sem_acento(v).replace("&", " and ").split()
    out = []
    for i, p in enumerate(pal):
        nua = re.sub(r"^[\W_]+|[\W_]+$", "", p)
        if (nua and nua.islower() and nua in LIG and nua not in COORD and i < len(pal) - 1
                and not re.search(r"\w[\W_]+\w", p)):
            continue
        out += [t.casefold() for t in re.split(r"[\W_]+", p) if t]
    return out


def singular(w):
    """Singular aproximado (pt/en/es), igual dos dois lados: so precisa ser consistente, nao gramatical."""
    if len(w) < 4 or re.search(r"\d", w) or _TAXON.search(w) or w.endswith(("ss", "us", "is", "ys")):
        return w
    for pl, sg in (("oes", "ao"), ("aes", "ao"), ("ais", "al"), ("eis", "el"), ("ies", "y"), ("ns", "m")):
        if w.endswith(pl):
            return w[:-len(pl)] + sg
    if w.endswith(("res", "zes", "les")) and len(w) > 5:
        return w[:-2]                                   # mulheres, cicatrizes, hospitales
    if w.endswith("s"):
        return w[:-1]
    return w


def chaves(v):
    """{nivel: chave} de um termo (digitacao usa a chave de 'ordem' depois de corrigir a palavra)."""
    c = conteudo(v)
    f = [singular(w) for w in c]
    return {"grafia": grafia(v), "ligacao": " ".join(c), "flexao": " ".join(f), "ordem": " ".join(sorted(f))}


def _apagadas(w):
    return {w[:i] + w[i + 1:] for i in range(len(w))}


class IndiceAproximado:
    """Indices de chaves -> descritores, montado uma vez a partir do Vocabulario."""

    def __init__(self, vocab):
        self.idx = {n: defaultdict(set) for n in NIVEIS[:4]}
        palavras = set()
        for ui, rec in vocab.base.items():
            if not vocab.caminhos(ui):
                continue
            for s in [rec["nome"], rec["en"]] + rec["sin_pt"] + rec["sin_en"] + rec["sin_es"]:
                if not s:
                    continue
                for n, k in chaves(s).items():
                    if k:
                        self.idx[n][k].add(ui)
                palavras.update(chaves(s)["flexao"].split())
        self.palavras = palavras
        # digitacao: indice de "uma letra apagada" (SymSpell, distancia 1) das palavras do DeCS com 7+ letras
        self.apagadas = defaultdict(set)
        for w in palavras:
            if len(w) >= 7 and w.isalpha():
                for d in _apagadas(w) | {w}:
                    self.apagadas[d].add(w)

    def _corrigir(self, w):
        """A palavra do DeCS a 1 letra de distancia de w (unica, mesmo comeco), ou None."""
        if len(w) < 7 or not w.isalpha() or w in self.palavras:
            return None
        cand = set()
        for d in _apagadas(w) | {w}:
            cand |= self.apagadas.get(d, set())
        cand = {c for c in cand if c != w and c[:3] == w[:3] and _distancia1(w, c)}
        return cand.pop() if len(cand) == 1 else None

    def uis(self, termo):
        """(uis, nivel) ou ([], None). So devolve casamento nao ambiguo (um descritor)."""
        ch = chaves(termo)
        for n in NIVEIS[:4]:
            achou = self.idx[n].get(ch[n])
            if achou:
                return (sorted(achou), n) if len(achou) == 1 else ([], None)
        pal = ch["flexao"].split()
        originais = {singular(grafia(t)): t for t in re.findall(r"\w+", termo)}
        for i, w in enumerate(pal):
            c = self._corrigir(w)
            # palavra que existe em portugues/ingles/espanhol nao e erro de digitacao ("castelo", "cinemateca")
            if c and not dicionario_tem(originais.get(w, w)):
                nova = pal[:i] + [c] + pal[i + 1:]
                achou = self.idx["ordem"].get(" ".join(sorted(nova)))
                if achou and len(achou) == 1:
                    return sorted(achou), "digitacao"
        return [], None


# dicionarios ortograficos do LibreOffice (os mesmos da clusterizacao das palavras-chave)
_DIC_PASTA = r"C:\Program Files\LibreOffice\share\extensions"
_DIC_ARQS = {"pt": (r"dict-pt-BR\pt_BR", r"dict-pt-PT\pt_PT"), "en": (r"dict-en\en_US", r"dict-en\en_GB"),
             "es": (r"dict-es\es_ES",)}
_DIC = None
_CACHE_DIC = {}


def dicionarios():
    """{idioma: [Dictionary]} (vazio se o LibreOffice/spylls nao estiver instalado)."""
    global _DIC
    if _DIC is None:
        try:
            from spylls.hunspell import Dictionary
            _DIC = {l: [Dictionary.from_files(os.path.join(_DIC_PASTA, a)) for a in arqs]
                    for l, arqs in _DIC_ARQS.items()}
        except Exception:
            _DIC = {}
    return _DIC


def dicionario_tem(palavra, lang=None):
    """A palavra existe no dicionario do idioma (lang=None: em algum de pt/en/es). Sem dicionarios: True."""
    ds = dicionarios()
    if not ds:
        return True
    k = (palavra, lang)
    if k not in _CACHE_DIC:
        _CACHE_DIC[k] = any(d.lookup(palavra) or d.lookup(palavra.lower()) or d.lookup(palavra.capitalize())
                            for l, dd in ds.items() if lang is None or l == lang for d in dd)
    return _CACHE_DIC[k]


def _distancia1(a, b):
    """Levenshtein <= 1 ou duas letras vizinhas trocadas."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        dif = [i for i in range(len(a)) if a[i] != b[i]]
        return len(dif) == 1 or (len(dif) == 2 and dif[1] == dif[0] + 1
                                 and a[dif[0]] == b[dif[1]] and a[dif[1]] == b[dif[0]])
    curto, longo = (a, b) if len(a) < len(b) else (b, a)
    return any(longo[:i] + longo[i + 1:] == curto for i in range(len(longo)))
