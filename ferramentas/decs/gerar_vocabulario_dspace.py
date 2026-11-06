# -*- coding: utf-8 -*-
"""Gera e CONFERE o decs.xml (vocabulario hierarquico do DSpace) — opcao D7.

FONTE UNICA. O gerador oficial e o script_conversao_decs_v2.py da pasta
"Vocabularios/DECS XML DSPACE" (o vocabulario "hibrido": arvore DeCS + notas
com escopo, termo em ingles, sinonimos, historico e anotacao, que aparecem na
interface do DSpace e orientam o indexador). Esta ferramenta NAO tem uma
segunda implementacao: ela importa aquele script e chama processar_lote_auto().
Assim o XML do projeto e o da pasta de vocabularios sao, por construcao, o mesmo.

Depois de gerar, confere:
  1. ids unicos (o DSpace localiza o no pelo id);
  2. filhos da raiz dentro de <isComposedBy>;
  3. nenhum no com o codigo como rotulo;
  4. os valores que o DSpace vai gravar (simulados como ele monta) sao
     IDENTICOS aos que o indexador usa (arca/decs_vocabulario.arvore_da_base);
  5. se o arquivo da pasta de vocabularios e identico byte a byte ao gerado aqui.
E escreve saidas/vocabulario_dspace/LEIA-ME.md com as diferencas em relacao ao
vocabulario publicado hoje no servidor.

Uso:  python ferramentas/decs/gerar_vocabulario_dspace.py [--todos-idiomas]
"""
import argparse
import filecmp
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from arca import config  # noqa: E402
from arca import decs_vocabulario as V  # noqa: E402

SAIDA_DIR = os.path.join(config.SAIDAS, "vocabulario_dspace")
SCRIPT_V2 = os.path.join(config.BIREME_DIR, "script_conversao_decs_v2.py")
PASTA_OFICIAL = os.path.join(config.BIREME_DIR, "vocabularios_dspace_v2")


def carregar_v2():
    if not os.path.exists(SCRIPT_V2):
        raise FileNotFoundError("Gerador oficial nao encontrado: %s" % SCRIPT_V2)
    spec = importlib.util.spec_from_file_location("script_conversao_decs_v2", SCRIPT_V2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def valores_como_dspace(raiz):
    """Simula o DSpace: {id: valor} percorrendo isComposedBy a partir da raiz."""
    saida = {}

    def andar(el, prefixo):
        comp = el.find("isComposedBy")
        if comp is None:
            return
        for no in comp.findall("node"):
            valor = (prefixo + "::" if prefixo else "") + no.get("label")
            saida.setdefault(no.get("id"), []).append(valor)
            andar(no, valor)

    andar(raiz, "")
    return saida


def conferir(caminho_xml, base):
    raiz = ET.parse(caminho_xml).getroot()
    problemas = []
    ids = Counter(n.get("id") for n in raiz.iter("node"))
    rep = [i for i, c in ids.items() if c > 1]
    if rep:
        problemas.append("%d ids repetidos (ex.: %s)" % (len(rep), rep[:3]))
    if raiz.find("isComposedBy") is None:
        problemas.append("filhos da raiz fora de <isComposedBy>")
    codigo = [n.get("id") for n in raiz.iter("node") if n.get("id") == n.get("label")]
    if codigo:
        problemas.append("%d nos com o codigo como rotulo: %s" % (len(codigo), codigo[:3]))
    valores = valores_como_dspace(raiz)
    esperado = V.arvore_da_base(base, com_categoria=True)
    diverg = [i for i, v in valores.items() if V.norm(esperado.get(i, "")) != V.norm(v[0])]
    faltam = [i for i in esperado if i not in valores]
    if diverg or faltam:
        problemas.append("%d valores diferentes e %d ausentes em relacao ao indexador (ex.: %s)"
                         % (len(diverg), len(faltam), (diverg + faltam)[:3]))
    return problemas, len(ids)


def relatorio_servidor(base, linhas):
    if not os.path.exists(config.VOCAB_SERVIDOR_JSON):
        return linhas
    serv = V.carregar_arvore_servidor()
    sem_cat = V.arvore_da_base(base, com_categoria=False)
    iguais = sum(1 for tn, v in serv.items() if V.chave_sem_categoria(v) == V.norm(sem_cat.get(tn, "")))
    so_servidor = sorted(tn for tn in serv if tn not in sem_cat and not tn.isalpha())
    so_novo = sorted(tn for tn in sem_cat if tn not in serv)
    mudou = sorted(tn for tn in serv if tn in sem_cat
                   and V.chave_sem_categoria(serv[tn]) != V.norm(sem_cat[tn]))
    proprios = [tn for tn in mudou
                if V.norm(serv[tn].split("::")[-1]) != V.norm(sem_cat[tn].split("::")[-1])]
    defeituosos = sorted(tn for tn, v in serv.items() if v.split("::")[-1].strip() == tn)
    linhas += ["## Diferencas em relacao ao vocabulario publicado hoje no servidor", "",
               "| | nos |", "|---|---|",
               "| iguais (mesmo numero de arvore e mesmo caminho) | %d |" % iguais,
               "| caminho mudou (o proprio nome ou o de um ancestral) | %d |" % len(mudou),
               "| ... dos quais o PROPRIO nome mudou | %d |" % len(proprios),
               "| so no servidor (sairam do DeCS 2026) | %d |" % len(so_servidor),
               "| so no DeCS 2026 (novos) | %d |" % len(so_novo), "",
               "Exemplos de nomes que mudaram no DeCS 2026:", ""]
    linhas += ["- `%s`: %s  ->  %s" % (tn, serv[tn].split("::")[-1], sem_cat[tn].split("::")[-1])
               for tn in proprios[:25]]
    if defeituosos:
        linhas += ["", "No defeituoso no vocabulario publicado hoje (rotulo = codigo; corrigido no v2.1):", ""]
        linhas += ["- `%s`: %s" % (tn, serv[tn]) for tn in defeituosos]
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todos-idiomas", action="store_true",
                    help="gera tambem en/es/fr (padrao: so pt, que e o publicado)")
    ap.add_argument("--so-conferir", action="store_true",
                    help="nao gera; confere o arquivo da pasta de vocabularios")
    args = ap.parse_args()
    idiomas = ("pt", "en", "es", "fr") if args.todos_idiomas else ("pt",)

    os.makedirs(SAIDA_DIR, exist_ok=True)
    oficial = os.path.join(PASTA_OFICIAL, "decs_hierarquico_pt.xml")
    if args.so_conferir:
        gerado = oficial
    else:
        carregar_v2().processar_lote_auto(config.BIREME_DIR, SAIDA_DIR, idiomas=idiomas)
        gerado = os.path.join(SAIDA_DIR, "decs_hierarquico_pt.xml")

    base = V.carregar_base()
    problemas, n = conferir(gerado, base)
    identico = (os.path.exists(oficial) and os.path.exists(gerado)
                and filecmp.cmp(gerado, oficial, shallow=False))

    linhas = ["# decs.xml - vocabulario hierarquico DeCS para o DSpace", "",
              "- Gerador: `script_conversao_decs_v2.py` (v2.1), pasta `%s`" % config.BIREME_DIR,
              "- Fonte: `%s`" % os.path.basename(config.BIREME_XML_POR),
              "- Arquivo conferido: `%s`" % gerado,
              "- Nos: %d" % n,
              "- Conferencias: %s" % ("todas OK" if not problemas else "; ".join(problemas)),
              "- Arquivo da pasta de vocabularios identico ao do projeto: **%s**"
              % ("sim" if identico else "NAO"), "",
              "Publicar no DSpace: copiar `decs_hierarquico_pt.xml` como "
              "`[dspace]/config/controlled-vocabularies/decs.xml`.", "",
              "Depois de publicar, rodar a opcao D6 (baixar vocabulario): o indexador passa a "
              "gravar no formato novo e converte os valores antigos.", ""]
    linhas = relatorio_servidor(base, linhas)
    with open(os.path.join(SAIDA_DIR, "LEIA-ME.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print("\n".join(linhas[:9]))
    if problemas:
        sys.exit(1)


if __name__ == "__main__":
    main()
