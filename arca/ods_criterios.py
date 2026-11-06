# -*- coding: utf-8 -*-
"""HISTORICO (prompt v2, 02-10/10/2026). NAO E O CRITERIO ATUAL: desde a D25 (10/10) e a D32 (28/10) o criterio
de ODS e so o texto oficial (lista dos 18 ODS + metas ONU/Brasil/conceitos Ipea 2026), sem regras por ODS.
O prompt v3 (FIXO em ods_professor.py) usa so OBJETIVOS daqui; o resto fica para reproduzir os rotulos v2.

Criterios de atribuicao de ODS (decididos com o responsavel em 2026-10-02).

Base: metas globais da ONU, metas brasileiras do Ipea ("Nacionalizacao da Agenda 2030", 2026;
antes, IPEA 2018) e glossarios da
ONU no Brasil (ODS 5, 6, 7, 9, 11, 12, 13 e 18), copiados em
dados/ods_referencia/. Este texto e usado (1) pelo LLM professor que rotula o
conjunto de treino e (2) como documentacao da regra. Mudou o criterio? Mude
aqui, incremente ods_config.VERSAO_ODS e rode a avaliacao de novo.
"""

OBJETIVOS_CURTOS = {
    1: "Erradicacao da pobreza; protecao social (1.3), acesso a servicos basicos e recursos (1.4), resiliencia dos pobres (1.5)",
    2: "Fome zero, seguranca alimentar e nutricional, ma-nutricao (2.1, 2.2), agricultura familiar e sustentavel, agroecologia (2.3, 2.4), recursos geneticos (2.5)",
    3: "Saude e bem-estar: mortalidade materna/infantil, doencas transmissiveis, negligenciadas e arboviroses (3.3), DCNT, saude mental e saude do trabalhador (3.4), drogas (3.5), transito, saude sexual e reprodutiva (3.7), SUS e acesso (3.8), poluicao (3.9), tabaco, P&D de tecnologias e inovacoes em saude (3.b), forca de trabalho em saude (3.c), vigilancia e alerta (3.d)",
    4: "Educacao de qualidade: educacao basica (4.1, 4.2), educacao profissional e superior (4.3), competencias tecnicas e profissionais (4.4), equidade na educacao (4.5), alfabetizacao (4.6), educacao para o desenvolvimento sustentavel, direitos humanos, cultura de paz (4.7), professores (4.c)",
    5: "Igualdade de genero: discriminacao de genero incluindo identidade de genero e orientacao sexual (5.1), violencia de genero (5.2), praticas nocivas (5.3), trabalho de cuidado nao remunerado (5.4), participacao (5.5), saude sexual e reprodutiva e direitos (5.6), recursos economicos (5.a), TIC e producao cientifica com perspectiva de genero (5.b)",
    6: "Agua potavel e saneamento, higiene, qualidade da agua e efluentes, gestao de recursos hidricos",
    7: "Energia acessivel, renovavel e eficiente",
    8: "Trabalho digno e crescimento: produtividade e inovacao (8.2), emprego e informalidade (8.3, 8.5), jovens (8.6), trabalho escravo e infantil (8.7), direitos e saude e seguranca no trabalho (8.8), turismo sustentavel (8.9)",
    9: "Industria, inovacao e infraestrutura: infraestrutura (9.1), atividade de alta complexidade tecnologica (9.2), pesquisa cientifica, capacidades tecnologicas e inovacao (9.5), desenvolvimento tecnologico nacional (9.b), acesso a TIC (9.c)",
    10: "Reducao das desigualdades: renda (10.1), inclusao de todos independentemente de idade, genero, deficiencia, raca, etnia, nacionalidade, religiao, condicao economica (10.2), igualdade de oportunidades e fim de praticas discriminatorias (10.3), protecao social e politica fiscal (10.4), migrantes e refugiados (10.7)",
    11: "Cidades e comunidades sustentaveis: moradia digna e assentamentos precarios/favelas (11.1), mobilidade (11.2), planejamento urbano participativo (11.3), patrimonio cultural e natural, inclusive acervos museologicos e documentais (11.4), desastres (11.5), qualidade do ar e residuos (11.6), espacos publicos (11.7)",
    12: "Consumo e producao responsaveis: uso eficiente de recursos (12.2), desperdicio de alimentos (12.3), manejo de produtos quimicos e residuos (12.4, 12.5), compras publicas sustentaveis (12.7), informacao e educacao ambiental (12.8)",
    13: "Acao contra a mudanca global do clima: adaptacao, resiliencia, educacao sobre clima",
    14: "Vida na agua: poluicao marinha, ecossistemas marinhos e costeiros, pesca, pesca artesanal (14.b)",
    15: "Vida terrestre: ecossistemas e biomas, florestas, biodiversidade e especies ameacadas (15.5), recursos geneticos e conhecimentos tradicionais (15.6), trafico de fauna, especies invasoras",
    16: "Paz, justica e instituicoes eficazes: violencia e homicidios (16.1), violencia contra criancas e adolescentes (16.2), acesso a justica (16.3), corrupcao, instituicoes transparentes e efetivas (16.6), participacao e conselhos (16.7), identidade civil (16.9), acesso a informacao e liberdades fundamentais (16.10), leis e politicas nao discriminatorias e afirmativas (16.b)",
    17: "Parcerias e meios de implementacao: cooperacao internacional e Sul-Sul, cooperacao em ciencia, tecnologia e inovacao (17.6), parcerias multissetoriais (17.16, 17.17), dados e estatisticas desagregados (17.18)",
    18: "Igualdade etnico-racial (Brasil): racismo e discriminacao contra povos indigenas e afrodescendentes no trabalho (18.1), violencia (18.2), justica (18.3), representatividade (18.4), reparacao, patrimonio cultural e religioso, cosmovisao (18.5), moradia, favelas e comunidades tradicionais (18.6), saude nao discriminatoria e saberes ancestrais (18.7), educacao e educacao antirracista, linguas indigenas (18.8), conhecimentos tradicionais e biodiversidade (18.9), imigrantes indigenas e afrodescendentes (18.10)",
}

CRITERIOS = """\
REGRAS (seguem as metas oficiais; na duvida, nao atribua):
1. Atribua um ODS quando o documento TRATA do tema de uma meta (objeto, populacao, problema ou
   resultado do estudo), nao quando o tema aparece so de passagem ou como contexto.
2. ODS 3: qualquer estudo de saude humana, doenca, servico ou politica de saude.
3. ODS 9 (decisao do responsavel, opcao 1a): SO quando o documento desenvolve, valida ou avalia uma
   tecnologia, produto, processo ou inovacao (software, aplicativo, plataforma, teste diagnostico,
   farmaco/vacina/adjuvante candidato, dispositivo, metodo ou ensaio novo, sistema de qualidade de
   laboratorio) ou trata de politica industrial/de inovacao. Base: meta 3.b Brasil (P&D de tecnologias e
   inovacoes em saude), correlata a 9.5. Pesquisa basica de mecanismo, morfologia, taxonomia ou
   epidemiologia descritiva NAO e ODS 9.
4. ODS 4: quando o objeto e educacao ou formacao: escola, ensino, professores, educacao em museus,
   formacao profissional e superior, inclusive de trabalhadores da saude (residencia, preceptoria,
   especializacao, educacao permanente, capacitacao). O documento ser tese/dissertacao NAO e motivo.
5. ODS 5: genero (inclui identidade de genero e orientacao sexual: discriminacao e violencia contra
   mulheres e pessoas LGBTQIA+), trabalho de cuidado, e saude sexual e reprodutiva tratada como
   direito, acesso, autonomia ou servico (contracepcao, aborto, pre-natal e parto como cuidado,
   aleitamento, violencia sexual ou obstetrica, prevencao de IST). Estudo clinico ou molecular de
   gestacao, cancer de mama ou HIV fica so no ODS 3.
6. ODS 10: o estudo analisa desigualdade, exclusao ou discriminacao (renda, raca, genero, idade,
   deficiencia, territorio, populacao de rua, privados de liberdade) como objeto ou resultado central.
7. ODS 18: populacao negra, indigena, quilombola, povos de terreiro ou comunidades tradicionais,
   racismo, saberes ancestrais, educacao antirracista. Vem junto com o ODS do tema (ex.: 3 + 18).
8. ODS 16: violencia (inclusive contra criancas), acesso a justica, direitos humanos e liberdades,
   instituicoes, participacao social e conselhos, acesso a informacao, leis nao discriminatorias.
9. ODS 11: moradia, favelas, cidade, mobilidade, desastres, patrimonio cultural (museus, acervos).
10. ODS 8: trabalho, emprego, renda do trabalho, saude e seguranca do trabalhador, trabalho infantil.
11. ODS 2: alimentacao, nutricao, seguranca alimentar, agricultura. ODS 6: agua e saneamento.
    ODS 15: biodiversidade terrestre e ecossistemas (nao o simples uso de animais em laboratorio).
    ODS 14: ambientes marinhos. ODS 12: produtos quimicos/agrotoxicos, residuos, educacao ambiental.
    ODS 17: cooperacao internacional, parcerias, diplomacia em saude, dados para o desenvolvimento.
12. Varios ODS sao comuns (ex.: 3 + 5 + 10 + 18). Mas nao atribua ODS sem base no texto.
13. Sem resumo e sem palavras-chave suficientes: decida pelo titulo; se nao der, lista vazia.

CUIDADOS (erros comuns a evitar):
- ODS 3 vale tambem para pesquisa de laboratorio, molecular, de patogenos, vetores, farmacos e
  diagnosticos: se o estudo e sobre saude ou doenca humana, inclua 3.
- ODS 18 SO se o texto fala explicitamente de populacao negra/preta/parda, indigenas, quilombolas,
  povos de terreiro ou de matriz africana, ciganos, ou de racismo. "Saber popular", "comunidade",
  "territorio", "vulnerabilidade" ou "favela" sozinhos NAO sao ODS 18.
- ODS 10 so quando desigualdade, discriminacao ou exclusao e objeto ou resultado central; uma
  mencao a "populacao vulneravel" ou a renda como variavel de ajuste nao basta.
- ODS 17 so com cooperacao internacional, parceria institucional ou diplomacia explicitas;
  compartilhar dados de pesquisa ou ciencia aberta sozinho NAO e 17.
- ODS 12 e 15 so quando o tema ambiental e o objeto (agrotoxico, residuo, biodiversidade, bioma);
  nao por usar animais, plantas ou "sustentavel" de passagem.
- ODS 9: inclua sempre que o estudo cria, desenvolve, valida ou avalia teste diagnostico, ensaio,
  metodo, kit, farmaco ou composto candidato, vacina, adjuvante, proteina recombinante, terapia nova,
  software, aplicativo, chatbot, plataforma, dispositivo, protocolo tecnologico ou sistema de gestao
  da qualidade de laboratorio.

EXEMPLOS (ilustrativos):
- "Validacao de RT-PCR em tempo real para deteccao de Leishmania" -> 3, 9
- "Atividade antiviral de derivados sinteticos contra Zika in vitro" -> 3, 9
- "Morfologia de ovos de triatomineos ao microscopio eletronico" -> 3
- "Mortalidade por COVID-19 segundo raca/cor no Brasil" -> 3, 10, 18
- "Residencia multiprofissional e formacao de preceptores no SUS" -> 3, 4
- "Violencia contra mulheres trans atendidas em servico de saude" -> 3, 5, 16
- "Saneamento e esquistossomose em municipios de baixa renda" -> 3, 6, 10
- "Diplomacia em saude e cooperacao Sul-Sul com paises africanos" -> 3, 17
- "Educacao popular em saude no territorio" -> 3 (e 4 se for formacao/curso)
"""


# ---------------------------------------------------------------------------
# Travas deterministicas (reforcam as regras 7 e 11 / CUIDADOS): um ODS que
# exige um tema explicito so fica se o texto contiver um termo desse tema.
# ---------------------------------------------------------------------------
import re as _re
import unicodedata as _ud


def _sem_acento(s):
    return "".join(c for c in _ud.normalize("NFD", s.lower()) if _ud.category(c) != "Mn")


EXIGE_TERMO = {
    18: _re.compile(r"\b(negr[oa]s?|pret[oa]s?|pard[oa]s?|raca|racial|raciais|racis\w*|etnic\w*|etnia\w*|"
                    r"indigena\w*|indio\w*|quilombo\w*|terreiro\w*|matriz africana|afro\w*|cigan\w*|"
                    r"black|race|racial|ethnic\w*|indigenous|quilombola\w*|afro-\w+|povos originarios|"
                    r"povos tradicionais|comunidades tradicionais|guarani|yanomami|kaingang|xavante)\b"),
    17: _re.compile(r"\b(cooperacao|internaciona\w*|sul-sul|diplomacia|parceria\w*|multilateral|bilateral|"
                    r"cooperation|international|partnership\w*|global health|saude global|paises|countries|"
                    r"onu|oms|opas|who|unicef|unesco|cplp|mercosul|africa\w*)\b"),
}


def travar(ods, texto):
    """FORA DO FLUXO desde a D24 (2026-10-10): o professor e soberano, nada e removido depois dele.
    Mantida so para reproduzir avaliacoes antigas (ferramentas/ods/avaliacao/*). Removia de `ods` os ODS
    que exigem termo explicito ausente do texto."""
    t = _sem_acento(texto or "")
    return [o for o in ods if o not in EXIGE_TERMO or EXIGE_TERMO[o].search(t)]


# ---------------------------------------------------------------------------
# Versao compacta para o prompt do professor (mesmas regras, menos tokens: o
# modelo hibrido reprocessa o prompt inteiro a cada documento).
# ---------------------------------------------------------------------------
OBJETIVOS_PROMPT = {
    1: "pobreza, protecao social", 2: "fome, nutricao, seguranca alimentar, agricultura familiar/agroecologia",
    3: "saude humana (qualquer doenca, servico, politica, pesquisa de saude, inclusive laboratorio)",
    4: "educacao e formacao (escola, ensino, professores, museus, formacao profissional e de trabalhadores da saude)",
    5: "genero, mulheres, LGBTQIA+, violencia de genero, cuidado nao remunerado, saude sexual e reprodutiva como direito/servico",
    6: "agua e saneamento", 7: "energia", 8: "trabalho, emprego, saude e seguranca do trabalhador, trabalho infantil",
    9: "desenvolvimento/validacao de tecnologia, produto, processo ou inovacao; politica industrial",
    10: "desigualdade, discriminacao, exclusao como tema central", 11: "cidade, moradia, favela, mobilidade, desastres, patrimonio/museus",
    12: "produtos quimicos/agrotoxicos, residuos, educacao ambiental", 13: "mudanca do clima", 14: "mar e zona costeira",
    15: "biodiversidade terrestre, ecossistemas, biomas", 16: "violencia, justica, direitos humanos, instituicoes, participacao social, acesso a informacao",
    17: "cooperacao internacional, parcerias, diplomacia em saude", 18: "populacao negra, indigena, quilombola, povos de terreiro, racismo",
}

# Prompt v2 (04/10 a 10/10/2026): metodo + regras por ODS + exemplos. Rotulou o acervo na rodada de 06-10/10.
# Desde a D25 o prompt tem so o metodo geral (CRITERIOS_PROMPT, abaixo); esta versao fica para reproducao.
CRITERIOS_PROMPT_V2 = """\
COMO DECIDIR (faca para cada meta candidata):
1. Leia o documento inteiro. Identifique o OBJETO do estudo, a POPULACAO, o PROBLEMA e o RESULTADO.
2. Para cada meta candidata, pergunte: o documento TRATA do que esta meta descreve (texto ONU ou texto Brasil)?
   Tratar = o tema da meta e objeto, populacao, problema ou resultado do estudo. Aparecer de passagem,
   no contexto, na justificativa ou como variavel de ajuste NAO conta.
3. Atribua o ODS da meta que melhor se encaixa e cite um trecho LITERAL do documento que comprove.
   Varios ODS sao comuns. Na duvida, nao atribua. Lista vazia e valida.
REGRAS POR ODS:
- 3: todo estudo de saude humana, inclusive de laboratorio, molecular, patogenos, vetores, farmacos.
- 9: so quando cria, desenvolve, valida ou avalia teste diagnostico, ensaio, metodo, kit, farmaco ou
  composto candidato, vacina, adjuvante, proteina recombinante, software, app, chatbot, plataforma,
  dispositivo ou sistema de qualidade. Pesquisa basica, morfologia, taxonomia, epidemiologia: nao.
- 4: objeto e educacao/formacao (inclui residencia, preceptoria, especializacao, capacitacao). Ser tese: nao.
- 5: genero, identidade de genero, orientacao sexual, violencia contra mulheres/LGBTQIA+; saude
  sexual e reprodutiva como direito, acesso ou servico (contracepcao, aborto, pre-natal, parto,
  aleitamento, violencia obstetrica, prevencao de IST). Estudo clinico/molecular de gestacao, cancer de mama ou HIV: so 3.
- 18: so com mencao explicita a negros/pretos/pardos, afrodescendentes, indigenas, quilombolas, povos e comunidades
  tradicionais, terreiro/matriz africana, ciganos, racismo ou raca/cor ANALISADA. Saude dessas populacoes = 18.7;
  moradia/favela dessas populacoes = 18.6. Favela, periferia, vulnerabilidade sozinhas: NAO e 18.
- 10: so se desigualdade/discriminacao e central. 17: so com cooperacao/parceria internacional explicita como
  objeto; coautoria internacional nao conta.
- 12 e 15: so se o tema ambiental e o objeto (uso de animais em laboratorio nao e 15).
EXEMPLOS:
- validacao de PCR para Leishmania -> 3 (3.3), 9 (9.5)
- morfologia de triatomineos -> 3 (3.3); NAO 9 nem 15
- mortalidade por COVID-19 segundo raca/cor -> 3 (3.3), 10 (10.2), 18 (18.7)
- residencia multiprofissional no SUS -> 3 (3.c), 4 (4.4)
- violencia contra mulheres trans -> 3, 5 (5.2), 16 (16.1)
- saneamento e esquistossomose -> 3 (3.3), 6 (6.2)
- cooperacao Sul-Sul em saude com paises africanos -> 3, 17 (17.9)
- grupos terapeuticos num CAPS AD (alcool e drogas) -> 3 (3.5)
- expressao genica do HIV em celulas -> so 3 (3.3); NAO 5
- urbanizacao de favela sem recorte racial -> 11 (11.1); NAO 18
- estudo com coautores de varios paises -> NAO 17 por isso
"""

# D25 (10/10/2026): o Gemma julga so pelas metas (texto ONU + Ipea 2026); sem regras por ODS nem exemplos.
CRITERIOS_PROMPT = CRITERIOS_PROMPT_V2[:CRITERIOS_PROMPT_V2.index("REGRAS POR ODS:")]
