# -*- coding: utf-8 -*-
"""Interpreta um valor de dc.subject.decs que ja esta no acervo.

Pergunta unica: "a que descritor DeCS este valor se refere?"

O acervo guarda o DECS de varias formas (medido em 2026-10-02, 156 mil valores):
    caminho valido no servidor  'Infeccoes::Zoonoses'                  ~9%
    nome do descritor           'Atencao Primaria a Saude'             ~74%
    sinonimo / nome antigo      'Saude do Trabalhador'                 ~7%
    com qualificador            'Brasil/epidemiologia'                 ~8%
    qualificador solto          'imunologia'                           <1%
    nada                        'Inovacao', 'Rio de Janeiro'           ~3%

Regras (nesta ordem):
1. Caminho valido no servidor: e o descritor daquele no. Nao se mexe.
2. Caminho que ja nao existe (renomeado em outra versao do DeCS): a folha
   diz o descritor; os niveis acima ajudam a escolher o galho.
3. Termo com qualificador: separa. Se a BIREME tem uma "combinacao de
   entrada" para o par (ex.: Gravidez/complicacoes -> Complicacoes na
   Gravidez), usa o descritor indicado — e a regra oficial de indexacao.
   O qualificador nao cabe na arvore do DSpace; fica registrado no backup
   [DECS-ORIGINAL] do item.
4. Nome preferido (PT, EN ou ES): descritor, com confianca total.
5. Sinonimo com um unico descritor: descritor (nome antigo, variante).
   Sinonimo com varios descritores: AMBIGUO — decide o documento.
6. Qualificador solto: nao e descritor; sai do campo (fica no backup).
7. Nada: NAO_ENCONTRADO — vai para revisao com candidatos.
"""
import re

from .decs_vocabulario import norm

# Tipos de resultado
CAMINHO_VALIDO = "caminho_valido"
CAMINHO_ANTIGO = "caminho_antigo"
NOME = "nome"
SINONIMO = "sinonimo"
COMBINACAO = "combinacao_oficial"
AMBIGUO = "ambiguo"
QUALIFICADOR_SOLTO = "qualificador_solto"
SEM_ARVORE = "sem_arvore"
NAO_ENCONTRADO = "nao_encontrado"

_RE_QUALIF = re.compile(r"^(?P<termo>.+?)\s*(?:/|\s-\s)\s*(?P<qual>[^/]+)$")


def _resultado(valor, tipo, ui=None, candidatos=(), qualificador="", caminho=None,
               confianca=1.0, nota=""):
    return {"valor": valor, "tipo": tipo, "ui": ui, "candidatos": list(candidatos),
            "qualificador": qualificador, "caminho_humano": caminho,
            "confianca": confianca, "nota": nota}


def _termo_simples(termo, vocab):
    """(tipo, ui, candidatos) para um termo sem caminho e sem qualificador."""
    uis, nivel = vocab.uis_do_termo(termo)
    if nivel == "nome" and len(uis) == 1:
        return NOME, uis[0], uis
    if nivel in ("sinonimo", "variante") and len(uis) == 1:
        return SINONIMO, uis[0], uis
    if uis:
        return AMBIGUO, None, uis
    return NAO_ENCONTRADO, None, []


def _galho_mais_parecido(vocab, ui, niveis_antigos):
    """Entre os caminhos do descritor, o que mais compartilha niveis com o antigo."""
    alvo = {norm(n) for n in niveis_antigos[:-1]}
    melhor, pontos = None, -1
    for c in vocab.caminhos(ui):
        p = len(alvo & {norm(n) for n in c.split("::")[:-1]})
        if p > pontos:
            melhor, pontos = c, p
    return melhor if pontos > 0 else None


def interpretar(valor, vocab):
    v = (valor or "").strip()
    if not v:
        return _resultado(valor, NAO_ENCONTRADO, nota="valor vazio")

    # 1-2. Caminho
    if "::" in v:
        canonico = vocab.valor_canonico(v)
        if canonico:
            return _resultado(valor, CAMINHO_VALIDO, vocab.ui_do_caminho(v), caminho=canonico)
        tn = vocab.arvore_do_caminho_em_outro_formato(v)
        if tn and vocab._tn_para_ui.get(tn):
            # mesmo no, so muda a presenca da categoria de topo: e o mesmo
            # galho que o catalogador escolheu, reescrito no formato atual.
            return _resultado(valor, CAMINHO_ANTIGO, vocab._tn_para_ui[tn],
                              caminho=vocab.servidor[tn], confianca=1.0,
                              nota="mesmo galho; formato da categoria de topo atualizado")
        niveis = [n.strip() for n in v.split("::") if n.strip()]
        tipo, ui, cands = _termo_simples(niveis[-1], vocab)
        if ui:
            galho = _galho_mais_parecido(vocab, ui, niveis)
            return _resultado(valor, CAMINHO_ANTIGO, ui, cands, caminho=galho,
                              confianca=0.95 if tipo == NOME else 0.85,
                              nota="caminho nao existe mais no vocabulario do servidor")
        return _resultado(valor, tipo if tipo == AMBIGUO else NAO_ENCONTRADO, None, cands,
                          nota="caminho invalido; folha '%s' nao identificada" % niveis[-1])

    # 3. Termo com qualificador.
    # Aceita qualificador fora da lista de 2026 quando a parte antes da barra e
    # um descritor conhecido: o acervo usa nomes antigos de qualificador
    # ("efeitos de drogas", "quimioterapia") que a BIREME renomeou.
    m = _RE_QUALIF.match(v)
    tipo_inteiro = _termo_simples(v, vocab)[0] if m else None
    if m and tipo_inteiro == NAO_ENCONTRADO and (
            norm(m.group("qual")) in vocab.qualificadores
            or _termo_simples(m.group("termo"), vocab)[1]):
        termo, qual = m.group("termo").strip(), m.group("qual").strip()
        tipo, ui, cands = _termo_simples(termo, vocab)
        if ui:
            q_ui = vocab.qualificadores.get(norm(qual))
            alvo = (vocab.base[ui].get("combina") or {}).get(q_ui) if q_ui else None
            if alvo and vocab.caminhos(alvo):
                return _resultado(valor, COMBINACAO, alvo, [alvo], qualificador=qual,
                                  nota="combinacao oficial DeCS: %s/%s -> %s"
                                  % (vocab.nome(ui), qual, vocab.nome(alvo)))
            permitido = norm(qual) in (vocab.base[ui].get("quals") or {})
            return _resultado(valor, tipo, ui, cands, qualificador=qual,
                              confianca=1.0 if tipo == NOME else 0.9,
                              nota="" if permitido else
                              "qualificador '%s' nao e permitido para este descritor" % qual)
        if norm(termo) in vocab.sem_arvore:
            ui = vocab.sem_arvore[norm(termo)]
            return _resultado(valor, SEM_ARVORE, ui, [ui], qualificador=qual,
                              nota="descritor DeCS sem posicao na arvore do servidor")
        return _resultado(valor, tipo, None, cands, qualificador=qual)

    # Qualificador solto escrito como qualificador (minusculo): no DeCS
    # descritor comeca com maiuscula e qualificador nao. "genetica" sozinho no
    # campo (138 vezes no acervo) e o qualificador de um termo vizinho, nao a
    # disciplina "Genetica"; "imunologia" nao e "Alergia e Imunologia".
    if v[:1].islower() and norm(v) in vocab.qualificadores:
        return _resultado(valor, QUALIFICADOR_SOLTO, qualificador=v,
                          nota="qualificador sem descritor; nao e assunto por si so")

    # 4-5. Termo simples
    tipo, ui, cands = _termo_simples(v, vocab)
    if tipo != NAO_ENCONTRADO:
        return _resultado(valor, tipo, ui, cands, confianca={NOME: 1.0, SINONIMO: 0.9}.get(tipo, 0.0))

    # 6. Descritor que existe no DeCS mas nao tem lugar na arvore do servidor
    if norm(v) in vocab.sem_arvore:
        ui = vocab.sem_arvore[norm(v)]
        return _resultado(valor, SEM_ARVORE, ui, [ui],
                          nota="descritor DeCS sem posicao na arvore do servidor "
                               "(check tag ou descritor ainda nao publicado)")

    # 7. Qualificador solto
    if norm(v) in vocab.qualificadores:
        return _resultado(valor, QUALIFICADOR_SOLTO, qualificador=v,
                          nota="qualificador sem descritor; nao e assunto por si so")

    return _resultado(valor, NAO_ENCONTRADO)
