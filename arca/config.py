# -*- coding: utf-8 -*-
"""Configuracao central: servidor, caminhos de arquivo e parametros medidos.

Tudo o que um humano precisaria mudar para replicar o projeto em outra
maquina ou outro repositorio esta aqui, e so aqui.
"""
import os

# Versao do projeto (vai para a provenance e para o SQLite).
VERSAO = "10.0.0"
# Versao do METODO de indexacao. Mudou o modelo, os sinais ou o limiar? Mude
# aqui: a opcao R1 lista os itens indexados com versao anterior para
# reprocessamento.
VERSAO_MODELO = "2026.10c-lexico+knn+rede+e5-gbm+raros(mistura+calibracao)+galho-decs2026"

# ---------------------------------------------------------------------------
# Servidor DSpace
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("DSPACE_API", "https://api.arca.fiocruz.br/api").rstrip("/")
DOMINIO_API = "api.arca.fiocruz.br"
HOSTNAME_UI = "arca.fiocruz.br"
VOCABULARIO_DSPACE = "decs"          # nome do vocabulario no DSpace
CAMPO_DECS = "dc.subject.decs"

# Login Microsoft (OIDC) do ARCA
OIDC_TENANT_ID = "bc3721a6-8061-4e19-a3b4-31e2093e209a"
OIDC_CLIENT_ID = "94a1103f-8656-4dbc-859d-c800eaea9eac"
OIDC_REDIRECT_URI = "https://api.arca.fiocruz.br/api/authn/oidc"
CDP_PORTA = os.getenv("CDP_PORTA", "9222")
CDP_ESPERA_LOGIN_S = 300
COOKIE_TTL_S = 3600

# ---------------------------------------------------------------------------
# MODO SOMENTE-LEITURA (opcional)
# ---------------------------------------------------------------------------
# Quando True, NENHUMA escrita sai para o ARCA: o cliente HTTP levanta excecao
# antes de enviar. Historico: ligado por padrao de 2026-10-02 ate a liberacao
# pelo responsavel pelo acervo no mesmo dia ("quero que o script rode real").
# Agora o padrao e GRAVAR (cada gravacao ainda pede confirmacao no menu).
# Para so simular: variavel de ambiente ARCA_SOMENTE_LEITURA=sim.
SOMENTE_LEITURA = os.getenv("ARCA_SOMENTE_LEITURA", "").strip().lower() == "sim"

HTTP_TIMEOUT = 120
HTTP_POOL = 32

# ---------------------------------------------------------------------------
# Pastas (tudo relativo a raiz do projeto)
# ---------------------------------------------------------------------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados")
# Dados GRANDES e regeneraveis (acervo coletado, embeddings, modelos treinados)
# ficam FORA da pasta do OneDrive: o OneDrive trava/corrompe arquivos de
# centenas de MB enquanto sao regravados (o acervo de 196 MB corrompeu em
# 2026-10-02). No OneDrive ficam so os pequenos e importantes (banco de
# rastreabilidade, vocabulario do servidor). Para mudar: ARCA_DADOS_LOCAIS.
DADOS_LOCAIS = os.getenv("ARCA_DADOS_LOCAIS",
                         os.path.join(os.getenv("LOCALAPPDATA", os.path.expanduser("~")),
                                      "CuradoriaARCA"))
CACHE = os.path.join(DADOS_LOCAIS, "cache")   # derivados: pode apagar, refaz sozinho
MODELOS = os.path.join(DADOS_LOCAIS, "modelos")
SAIDAS = os.path.join(RAIZ, "saidas")
SAIDAS_DECS = os.path.join(SAIDAS, "decs")
SAIDAS_IDIOMAS = os.path.join(SAIDAS, "idiomas")
SAIDAS_AVALIACAO = os.path.join(SAIDAS, "avaliacao")
LOGS = os.path.join(RAIZ, "logs")

for _p in (DADOS, DADOS_LOCAIS, CACHE, MODELOS, SAIDAS_DECS, SAIDAS_IDIOMAS, SAIDAS_AVALIACAO, LOGS):
    os.makedirs(_p, exist_ok=True)

COOKIES_JSON = os.path.join(DADOS, "sessao", "cookies_arca.json")
PERFIL_NAVEGADOR = os.path.join(DADOS, "sessao", "perfil_navegador")

# Fontes do vocabulario
BIREME_DIR = _BIREME_DIR = os.getenv(
    "DECS_BIREME_DIR",
    os.path.join("dados", "bireme"))
BIREME_XML_POR = os.getenv("DECS_BIREME_POR", os.path.join(_BIREME_DIR, "bireme_decs_por2026.xml"))
BIREME_XML_SPA = os.getenv("DECS_BIREME_SPA", os.path.join(_BIREME_DIR, "bireme_decs_spa2026.xml"))
VOCAB_SERVIDOR_JSON = os.path.join(DADOS, "vocabulario_servidor_decs.json")
DECS_BASE_JSON = os.path.join(CACHE, "decs_base.json")

# ---------------------------------------------------------------------------
# Formato da arvore DeCS no DSpace
# ---------------------------------------------------------------------------
# Incluir as 21 grandes categorias (ANATOMIA, DOENCAS, SAUDE PUBLICA...) como
# primeiro nivel da arvore. Decisao do responsavel (2026-10-02): navegacao e
# filtros comecam por 21 opcoes em vez de ~140 — e o que os navegadores
# oficiais do DeCS/MeSH fazem. O valor gravado fica
# "DOENCAS [C]::Infeccoes::Zoonoses".
# ATENCAO: so tem efeito na GRAVACAO depois que o decs.xml gerado com esta
# opcao estiver publicado no DSpace; ate la o script grava no formato que o
# servidor reconhece (o vocabulario baixado do servidor manda).
DECS_INCLUIR_CATEGORIA = True

# Em que formato GRAVAR o dc.subject.decs:
#   "xml_v2"   -> a arvore do decs.xml gerado pelo script_conversao_decs_v2.py
#                 (DeCS 2026, com as 21 categorias). Decisao do responsavel
#                 (2026-10-02): o metadado e armazenado ja com as categorias.
#                 Enquanto o XML novo nao estiver publicado no DSpace, os
#                 valores nao aparecem ligados a arvore da interface atual;
#                 depois da publicacao, batem exatamente (a D7 confere).
#   "servidor" -> a arvore publicada hoje no DSpace (copia baixada pela D6).
DECS_ALVO_GRAVACAO = os.getenv("DECS_ALVO_GRAVACAO", "xml_v2")
# Rotulo da categoria: {nome} (ex.: "Doencas"), {NOME} (maiusculas), {cod} ("C")
DECS_ROTULO_CATEGORIA = "{NOME} [{cod}]"
DECS_CATEGORIAS = {
    "A": "Anatomia", "B": "Organismos", "C": "Doenças",
    "D": "Compostos Químicos e Drogas",
    "E": "Técnicas e Equipamentos Analíticos, Diagnósticos e Terapêuticos",
    "F": "Psiquiatria e Psicologia", "G": "Fenômenos e Processos",
    "H": "Disciplinas e Ocupações", "I": "Humanidades",
    "J": "Tecnologia, Indústria e Agricultura", "K": "Ciências Humanas",
    "L": "Ciência da Informação", "M": "Denominações de Grupos",
    "N": "Assistência à Saúde", "V": "Características de Publicações",
    "Z": "Denominações Geográficas", "HP": "Homeopatia",
    "MT": "Medicinas Tradicionais, Complementares e Integrativas",
    "SH": "Ciência e Saúde", "SP": "Saúde Pública", "VS": "Vigilância Sanitária",
}


def rotulo_categoria(cod):
    nome = DECS_CATEGORIAS.get(cod, cod)
    return DECS_ROTULO_CATEGORIA.format(nome=nome, NOME=nome.upper(), cod=cod)


# ---------------------------------------------------------------------------
# Politica de atribuicao de DECS (os limiares numericos saem da avaliacao)
# ---------------------------------------------------------------------------
# Precisao MEDIDA exigida para gravar um descritor novo sem revisao humana,
# e para coloca-lo na lista de revisao. O limiar de probabilidade que atinge
# cada precisao e calculado por ferramentas/decs/treinar_avaliar.py.
# Modelo de texto (ver arca/decs_embeddings.py: MODELOS). A escolha sai da
# avaliacao (D5), nao do ranking geral: ver docs/DECISOES.md.
MODELO_EMBEDDING = os.getenv("DECS_MODELO_EMBEDDING", "e5-large")

DECS_PRECISAO_AUTOMATICA = 0.90
# 0,60 gerava ~2,5 sugestoes por item (175 mil no acervo): lista grande demais
# para alguem revisar. 0,70 deixa a lista menor e mais confiavel.
DECS_PRECISAO_REVISAO = 0.70
# Opcoes de melhoria do DeCS (decididas em 05/10; DESLIGADAS ate serem medidas em ferramentas/decs/treinar_avaliar.py):
# peso de qualidade dos exemplos humanos no treino do classificador, e segunda opiniao do LLM local
# (arca/decs_segunda_opiniao.py) na faixa de revisao humana.
DECS_PESO_QUALIDADE = os.getenv("DECS_PESO_QUALIDADE", "nao") == "sim"
DECS_SEGUNDA_OPINIAO = os.getenv("DECS_SEGUNDA_OPINIAO", "nao") == "sim"
# Palavra-chave do autor -> DeCS por grafia/ligacao/plural/ordem/digitacao (arca/decs_palavra_chave.py, pedido de
# 26/10). Acrescenta a caracteristica lex_kw_aprox: ligar exige retreinar (ferramentas/decs/treinar_avaliar.py).
DECS_KW_APROXIMADO = os.getenv("DECS_KW_APROXIMADO", "nao") == "sim"
DECS_MAX_NOVOS_POR_ITEM = 5
# Descritores que o CLASSIFICADOR nao sugere ate a calibracao ser corrigida e medida (amostra de 2026-10-20:
# "Politica de Saude" em 84% dos termos do modelo e "Eugenia" com p=0,999 sem relacao; DECISOES §18). Vindos da
# palavra-chave do autor continuam valendo.
# "Eugenia" saiu do bloqueio em 2026-10-25 (pedido do responsavel: bloqueava tambem a indexacao correta; DECISOES §19).
DECS_CLASSIFICADOR_BLOQUEADOS = {"Política de Saúde"}
# Passo 3 dos raros (arca/decs_raros.py): antes de gravar SOZINHO um descritor com menos de decs_raros.K_LLM exemplos
# humanos, o LLM local le o documento e a definicao do descritor e confirma; sem confirmacao vira sugestao.
# DESLIGADO ate ser medido em ferramentas/decs/avaliar_llm_raros.py (regra 8).
DECS_LLM_RAROS = os.getenv("DECS_LLM_RAROS", "nao") == "sim"
# Termo do catalogador que NAO e DeCS ("Inovacao", "Rio de Janeiro"):
#   "palavra_chave" -> sai do campo de DECS e vai para CAMPO_PALAVRA_CHAVE
#   "manter"        -> fica no campo de DECS e vai para a revisao
DECS_TERMO_FORA_DO_DECS = "palavra_chave"
CAMPO_PALAVRA_CHAVE = "dc.subject.other"

# Acervo coletado (D4) e rastreabilidade
ACERVO_JSON = os.path.join(DADOS_LOCAIS, "acervo_arca.json")
BANCO_SQLITE = os.path.join(DADOS, "decs_indexacao.db")
