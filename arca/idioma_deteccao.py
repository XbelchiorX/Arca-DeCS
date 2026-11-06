# -*- coding: utf-8 -*-
"""Deteccao do idioma de titulos e textos curtos do ARCA.

Problemas medidos no detector anterior (2.809 discordancias em 92.786 titulos;
170 rotuladas a mao em 2026-10-02 — ver docs/DECISOES.md):
  - NOMES CIENTIFICOS em latim puxam o titulo para es/it/fr/tr
    ("Thelazia callipaeda" -> es; titulos ingleses cheios de taxons -> it).
  - LUGARES brasileiros puxam titulo ingles para pt ("... in Sao Paulo").
  - PORTUGUES EM GRAFIA ANTIGA, sem acento ("Sobre a sistematica dos
    tabanideos", "Estudos sobre a Variola") vira espanhol.
  - 12 idiomas candidatos (incluindo japones, turco, arabe) para um acervo que
    so tem pt, en, es, fr, de, it: mais candidatos = mais confusao.

O que este detector faz:
  1. LIMPA o texto: tira citacoes entre aspas, binomios e generos (lista de organismos do proprio
     DeCS), autoria taxonomica ("Lent & Wygodzinsky, 1979"), parenteses de
     classificacao ("(Hemiptera: Reduviidae)"), "n. sp.", e nomes de lugares
     brasileiros.
  2. Pergunta ao lingua so entre os 6 idiomas do acervo.
  3. Soma evidencia de PALAVRAS EXCLUSIVAS de cada lingua ("dos", "the",
     "del", "und"...). Palavra que existe em duas linguas ("para", "sobre")
     nao conta.
  4. ABSTEM-SE quando nao ha seguranca (texto curto demais depois da limpeza
     ou empate): devolve None, e quem chama nao altera nada.
"""
import re

_IDIOMAS = ("pt-BR", "en", "es", "fr", "de", "it")

_EXCLUSIVAS = {
    "pt-BR": {"os", "as", "dos", "das", "do", "da", "no", "na", "nos", "nas", "um", "uma",
              "ao", "aos", "à", "às", "não", "com", "pelo", "pela", "pelos", "pelas", "em",
              "sôbre", "uns", "umas", "também", "ou", "seu", "sua", "seus", "suas"},
    "en": {"the", "of", "and", "with", "on", "from", "by", "is", "are", "among", "between",
           "in", "its", "their", "after", "during", "using", "toward", "towards", "new"},
    "es": {"el", "los", "las", "del", "y", "en", "una", "con", "al", "es", "lo", "su", "sus",
           "entre los", "según", "también", "o"},
    "fr": {"le", "les", "des", "du", "et", "sur", "une", "au", "aux", "dans", "chez", "pour",
           "est", "ou"},
    "de": {"der", "die", "das", "und", "von", "mit", "über", "ueber", "zur", "zum", "den",
           "dem", "ein", "eine", "des", "im"},
    "it": {"il", "gli", "della", "delle", "dei", "degli", "nel", "nella", "di", "per", "lo"},
}
# palavras que aparecem em mais de uma lista nao servem de evidencia
_conta = {}
for _l, _ws in _EXCLUSIVAS.items():
    for _w in _ws:
        _conta[_w] = _conta.get(_w, 0) + 1
_EXCLUSIVAS = {l: {w for w in ws if _conta[w] == 1} for l, ws in _EXCLUSIVAS.items()}

_LUGARES = [
    "rio de janeiro", "são paulo", "sao paulo", "minas gerais", "espírito santo", "espirito santo",
    "rio grande do sul", "rio grande do norte", "mato grosso do sul", "mato grosso", "santa catarina",
    "distrito federal", "são gonçalo", "são joão", "são francisco", "são luís", "belo horizonte",
    "porto alegre", "salvador", "recife", "fortaleza", "manaus", "belém", "curitiba", "goiânia",
    "teresina", "natal", "maceió", "aracaju", "joão pessoa", "florianópolis", "vitória", "cuiabá",
    "campo grande", "porto velho", "rio branco", "macapá", "boa vista", "palmas", "são luis",
    "bahia", "pernambuco", "maranhão", "piauí", "ceará", "paraíba", "alagoas", "sergipe", "pará",
    "amapá", "amazonas", "roraima", "rondônia", "acre", "tocantins", "goiás", "paraná",
    "fiocruz", "instituto oswaldo cruz", "oswaldo cruz", "manguinhos", "amazônia", "amazonia",
    "pantanal", "nordeste", "sudeste", "mata atlântica",
]
_RE_LUGARES = re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(_LUGARES, key=len, reverse=True))
                         + r")\b", re.IGNORECASE)
_RE_AUTORIA = re.compile(r"\b[A-Z][\w'\-]+(?:\s*(?:&|e|and|et)\s*[A-Z][\w'\-]+)*,?\s*\(?\d{4}\)?")
_RE_PARENTESE_TAXON = re.compile(r"\((?:[^()]*[A-Z][a-z]+(?:idae|inae|ptera|oda|ales|aceae|ida|ina)[^()]*)\)")
_RE_ABREV = re.compile(r"\b(?:n\.\s*sp|sp\.\s*n|sp|spp|n\.\s*gen|gen\.\s*n|nov|sp\.\s*nov|syn|cf|aff)\.?(?=\W|$)",
                       re.IGNORECASE)
# Texto entre aspas e nome de obra/evento citado, nao o idioma do titulo:
# 'Documentary "Vigilancia Popular frente as emergencias..."' e a versao
# INGLESA de um titulo da RECIIS (medido em 2026-10-02).
_RE_CITACAO = re.compile(r"[“”\"«][^“”\"«»]{3,}[“”\"»]")
_RE_PALAVRA = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)

_detector = {}
_generos = set()


def configurar_generos(nomes_de_organismos):
    """Recebe nomes de organismos (ex.: do DeCS, arvore B) e guarda os generos.

    Genero = 1a palavra de um nome binomial ("Trypanosoma cruzi" -> trypanosoma).
    """
    for nome in nomes_de_organismos:
        partes = (nome or "").split()
        if len(partes) >= 2 and partes[0][:1].isupper() and partes[1][:1].islower() \
                and partes[0].isalpha() and len(partes[0]) > 3:
            _generos.add(partes[0].lower())


def _carregar_generos_do_decs():
    if _generos:
        return
    try:
        from . import decs_vocabulario as V
        base = V.carregar_base()
        nomes = []
        for rec in base.values():
            if any(tn.startswith("B") for tn in rec["arvores"]):
                nomes += [rec["nome"], rec["en"]] + rec["sin_en"] + rec["sin_pt"]
        configurar_generos(nomes)
    except Exception:
        pass        # sem a base DeCS, segue sem a lista (so as outras limpezas)


def limpar(texto):
    """Tira do texto o que nao e lingua natural: taxons, autoria, lugares."""
    _carregar_generos_do_decs()
    t = texto or ""
    t = _RE_CITACAO.sub(" ", t)
    t = _RE_PARENTESE_TAXON.sub(" ", t)
    t = _RE_AUTORIA.sub(" ", t)
    t = _RE_ABREV.sub(" ", t)
    t = _RE_LUGARES.sub(" ", t)
    palavras = _RE_PALAVRA.findall(t)
    saida = []
    pula_epiteto = False
    for p in palavras:
        pl = p.lower()
        if pl in _generos:
            pula_epiteto = True
            continue
        if pula_epiteto and p[:1].islower() and pl not in _conta:
            pula_epiteto = False       # epiteto especifico ("cruzi", "mansoni")
            continue
        pula_epiteto = False
        saida.append(p)
    return " ".join(saida)


def _lingua():
    if "d" not in _detector:
        from lingua import Language, LanguageDetectorBuilder
        mapa = {"pt-BR": Language.PORTUGUESE, "en": Language.ENGLISH, "es": Language.SPANISH,
                "fr": Language.FRENCH, "de": Language.GERMAN, "it": Language.ITALIAN}
        _detector["d"] = LanguageDetectorBuilder.from_languages(*mapa.values()).build()
        _detector["mapa"] = {v.name: k for k, v in mapa.items()}
    return _detector["d"], _detector["mapa"]


MIN_PALAVRAS = 3
JANELA = 6
PASSO = 3
PESO_EXCLUSIVA = 0.25
MARGEM_MINIMA = 0.15


def pontuar(texto):
    """{idioma: pontuacao} para o texto limpo (lingua + palavras exclusivas)."""
    limpo = limpar(texto)
    palavras = [p.lower() for p in _RE_PALAVRA.findall(limpo)]
    det, mapa = _lingua()
    pontos = {l: 0.0 for l in _IDIOMAS}
    # O lingua tem regras internas que, ao ver UMA palavra, cravam 100% num
    # idioma (medido: "... of tuberculosis patients with or without HIV
    # co-infection" = ingles 0,98; + " in a reference hospital" = espanhol
    # 1,00). Media sobre janelas de JANELA palavras: uma palavra so nao decide.
    janelas = [palavras[i:i + JANELA] for i in range(0, max(1, len(palavras) - JANELA + 1), PASSO)]
    if len(palavras) > JANELA and (len(palavras) - JANELA) % PASSO:
        janelas.append(palavras[-JANELA:])
    janelas = [j for j in janelas if j]
    for j in janelas:
        for c in det.compute_language_confidence_values(" ".join(j)):
            pontos[mapa[c.language.name]] += c.value / len(janelas)
    for l, ws in _EXCLUSIVAS.items():
        n = sum(1 for p in palavras if p in ws)
        pontos[l] += PESO_EXCLUSIVA * min(n, 3)
    return pontos, len(palavras)


def detectar(texto):
    """(idioma, margem) ou (None, margem) quando nao ha seguranca."""
    pontos, n = pontuar(texto)
    ordem = sorted(pontos.items(), key=lambda kv: -kv[1])
    (l1, p1), (_l2, p2) = ordem[0], ordem[1]
    margem = p1 - p2
    if n < MIN_PALAVRAS or margem < MARGEM_MINIMA:
        return None, margem
    return l1, margem
