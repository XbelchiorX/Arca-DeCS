# -*- coding: utf-8 -*-
"""LLM local "professor": rotula ODS para treinar o classificador final.

Modelo (decisao D19, 2026-10-04): Gemma 4 26B-A4B (Google, Apache-2.0), GGUF oficial QAT q4_0, via
llama-server; parte dos "especialistas" (MoE) fica na RAM porque o modelo (14,4 GB) nao cabe nos 8 GB.
Ajuste medido (04/10): --n-cpu-moe 24 e lote 2048 = 5,7 s/item (99/512: 12,2 s; abaixo de 24 estoura a GPU: ~30 s).
No gabarito cego: F1 0,822 (sem ODS 3: 0,718) contra 0,759/0,647 do Qwen3.5-9B com os mesmos prompts.
ODS_PROFESSOR=qwen3.5-9b volta ao anterior. Professor LOCAL, sem API paga (D16).
Os prompts sao montados no formato ChatML e enviados como mensagens (system/user) ao
/v1/chat/completions, que aplica o modelo de conversa do proprio modelo.

Entrada: titulo + resumo completo + todas as palavras-chave, os criterios
oficiais (ods_criterios) e as 20 metas mais proximas (ods_metabusca), que
cobrem 91% dos ODS do gabarito; os 18 objetivos resumidos vao sempre no
prompt, entao o professor pode escolher um ODS fora das candidatas.
Saida: lista de {ods, meta, trecho}. As respostas ficam em cache (JSON por
handle) para nao repetir trabalho.

O prompt comeca por uma parte fixa (papel + criterios + objetivos) e so
depois vem o documento: o llama.cpp reaproveita a parte fixa ja processada.
"""
import glob
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from . import config
from . import ods_metabusca as MB
from .ods_criterios import CRITERIOS_PROMPT_V2, OBJETIVOS_PROMPT
from .ods_metas import OBJETIVOS

MODELOS = {
    "gemma4-26b": {"repo": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "arq": "gemma-4-26B_q4_0-it.gguf",
                   "nome": "gemma-4-26B-A4B-it-qat-q4_0", "paralelo": 2,
                   "args": ["--n-cpu-moe", os.getenv("ODS_GEMMA_CPU_MOE", "24"), "-b", os.getenv("ODS_GEMMA_UB", "2048"),
                            "-ub", os.getenv("ODS_GEMMA_UB", "2048")]},
    "qwen3.5-9b": {"repo": "unsloth/Qwen3.5-9B-GGUF", "arq": "Qwen3.5-9B-Q4_K_M.gguf",
                   "nome": "Qwen3.5-9B-Q4_K_M", "paralelo": 2, "args": []},
    # 7,0 GB: nao cabe inteiro nos 8 GB com contexto; ngl None = o llama.cpp ajusta sozinho quanto vai para a GPU
    "gemma4-12b": {"repo": "google/gemma-4-12B-it-qat-q4_0-gguf", "arq": "gemma-4-12b-it-qat-q4_0.gguf",
                   "nome": "gemma-4-12b-it-qat-q4_0", "paralelo": 1, "ngl": None, "args": []},
    # candidatos de 28/10 (pedido do responsavel): MoE como o Gemma 26B, especialistas na RAM. Medido em 10 itens
    # com lote 2048: Qwen 40 na CPU 8,7 s/item (30: 46 s, estoura a GPU); gpt-oss 24: 15,6 s (18: 16,8; 6: 33,7)
    "qwen3.5-35b": {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "arq": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
                    "nome": "Qwen3.5-35B-A3B-Q4_K_M", "paralelo": 2,
                    "args": ["--n-cpu-moe", os.getenv("ODS_CPU_MOE", "35"), "-b", os.getenv("ODS_UB", "2048"),
                             "-ub", os.getenv("ODS_UB", "2048")]},
    # 02/11 (pedido da responsavel: base da capacidade dos menores): Gemma 4 E4B e E2B, Apache-2.0, cabem na GPU
    "gemma4-e4b": {"repo": "google/gemma-4-E4B-it-qat-q4_0-gguf", "arq": "gemma-4-E4B_q4_0-it.gguf",
                   "nome": "gemma-4-E4B-it-qat-q4_0", "paralelo": 1, "args": ["-ctk", "q8_0", "-ctv", "q8_0"]},
    "gemma4-e2b": {"repo": "google/gemma-4-E2B-it-qat-q4_0-gguf", "arq": "gemma-4-E2B_q4_0-it.gguf",
                   "nome": "gemma-4-E2B-it-qat-q4_0", "paralelo": 1, "args": ["-ctk", "q8_0", "-ctv", "q8_0"]},
    # 02/11 (pedido da responsavel: "o modelo mais popular do Llama"): Llama-3.1-8B-Instruct, o mais baixado da Meta
    # no Hugging Face; licenca Llama 3.1 Community; 4,9 GB. Com 2 vagas de 20k tokens o contexto (f16) nao cabia junto
    # nos 8 GB e ia para a CPU (2,9 tokens/s): 1 vaga e contexto em q8_0
    "llama3.1-8b": {"repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "arq": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    "nome": "Meta-Llama-3.1-8B-Instruct-Q4_K_M", "paralelo": 1, "args": ["-ctk", "q8_0", "-ctv", "q8_0"]},
    # 02/11 (pedido da responsavel: "mudar o Qwen para a versao 3.8"): Qwen3.8-27B, denso (os 27B trabalham em toda
    # palavra), Apache-2.0, 16,5 GB; nao cabe nos 8 GB: o llama.cpp poe na GPU o que couber (ngl None), o resto na CPU
    "qwen3.8-27b": {"repo": "unsloth/Qwen3.8-27B-GGUF", "arq": "Qwen3.8-27B-UD-Q4_K_M.gguf",
                    "nome": "Qwen3.8-27B-UD-Q4_K_M", "paralelo": 1, "ngl": None, "args": []},
    # 02/11 (pedido da responsavel): Qwen3.8-Flash-Next, 125B MoE (6B ativos), licenca Qwen Community 1.0. IQ3_XXS
    # (82 GB) no NAS; nao cabe na RAM (32 GB): todos os especialistas fora da GPU, o resto lido do disco (mmap).
    "qwen3.8-flash-next": {"repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
                           "arq": "UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
                           "nome": "Qwen3.8-Flash-Next-UD-IQ3_XXS", "paralelo": 1,
                           "args": ["--n-cpu-moe", os.getenv("ODS_CPU_MOE", "999"), "-b", os.getenv("ODS_UB", "2048"),
                                    "-ub", os.getenv("ODS_UB", "2048")]},
    # gpt-oss raciocina antes de responder (formato harmony); esforco baixo e mais tokens de saida
    "gpt-oss-20b": {"repo": "ggml-org/gpt-oss-20b-GGUF", "arq": "gpt-oss-20b-MXFP4.gguf",
                    "nome": "gpt-oss-20b-MXFP4", "paralelo": 2, "kwargs": {"reasoning_effort": "low"},
                    "n_predict": 1500, "args": ["--n-cpu-moe", os.getenv("ODS_CPU_MOE", "24"), "-b", os.getenv("ODS_UB", "2048"),
                                                 "-ub", os.getenv("ODS_UB", "2048")]},
}
PROFESSOR = os.getenv("ODS_PROFESSOR", "qwen3.5-35b")   # D31 (29/10): Qwen3.5-35B-A3B, professor individual
_CFG = MODELOS[PROFESSOR]
GGUF = os.getenv("ODS_PROFESSOR_GGUF", "")
NAS_LLM = os.getenv("ODS_NAS_LLM", r"\\SEU_NAS\compartilhamento\LLM")   # modelos guardados (D41)
REPO, ARQUIVO = _CFG["repo"], _CFG["arq"]
NOME = _CFG["nome"]
PROMPT_VERSAO = int(os.getenv("ODS_PROMPT", "3"))   # 29/10: v4 medido e rejeitado (D34), volta o v3
# 4 (28/10, D34): o mesmo texto do rotulador do gabarito v7, com "metas candidatas"; ODS_PROMPT=3 reproduz o v3
# 3 (10/10, D25): so as metas e o metodo geral, sem regras por ODS nem exemplos
# 2 (04/10): + regras por ODS e exemplos. Rotulos v2 continuam validos (D25: o acervo nao e refeito)
PROMPTS_ACEITOS = (PROMPT_VERSAO,)   # D27 (20/10): os rotulos v2 nao valem mais; D34: so os da versao em uso
# Um arquivo de rotulos por versao do prompt e professor (D28): Qwen e Gemma rotulam o acervo sem um apagar o outro,
# para a votacao. Os 69.178 rotulos v2 do Gemma continuam em ods_professor.json (historico, nao usados).
CACHE_JSON = os.path.join(config.CACHE, "ods_professor_v%d_%s.json" % (PROMPT_VERSAO, NOME))
CACHE_JSON_V2 = os.path.join(config.CACHE, "ods_professor.json")
# D28: a rodada do acervo usa o prompt v3 COM o trecho literal (50 itens, 21/10: F1 sem ODS 3 0,583 x 0,567 no
# Gemma e 0,571 x 0,511 no Qwen; custa ~1 s/item)
ROTULAR_COM_TRECHO = os.getenv("ODS_COM_TRECHO", "sim") != "nao"
# Servidor llama.cpp com CUDA (muito mais rapido e com 4 documentos em paralelo).
# Se ODS_LLM_URL estiver vazio, usa o llama-cpp-python dentro do processo.
SERVIDOR = os.getenv("ODS_LLM_URL", "http://127.0.0.1:8089")
CHAVE_ARQ = os.path.join(os.path.expanduser("~"), ".ods_llm_key")
PARALELO = int(os.getenv("ODS_LLM_PARALELO", str(_CFG["paralelo"])))   # Qwen: 3 vagas x 12k nos 8 GB; Gemma: 2
CONTEXTO = int(os.getenv("ODS_LLM_CONTEXTO", "20480"))  # tokens por vaga; maior prompt do acervo (documento + 20 metas ONU+Ipea 2026 inteiras + resumo do PDF) cabe; o servidor recusa o que passar
_M = {}

_JSON = ('{"ods": [{"n": 3, "meta": "3.3", "trecho": "ate 20 palavras copiadas do documento que comprovam"}]}\n'
         'Lista vazia se nenhum ODS se aplica: {"ods": []}<|im_end|>\n')

# Prompt v3 (D25): so textos oficiais (titulos dos 18 ODS e as metas ONU + Ipea 2026) e a tarefa. Nada mais.
# Experimento de 22/10 (pedido do responsavel): ODS_SEM_LISTA=sim tira a lista dos 18 ODS do prompt (so as metas).
SEM_LISTA = os.getenv("ODS_SEM_LISTA", "nao") == "sim"
FIXO = (
    "<|im_start|>system\nVoce indexa producao cientifica pelos Objetivos de Desenvolvimento Sustentavel (ODS): "
    "metas globais da ONU e metas brasileiras do Ipea (2026, Nacionalizacao da Agenda 2030), incluindo o ODS 18.\n\n" +
    ("" if SEM_LISTA else "OS 18 ODS (titulos oficiais):\n" + "\n".join("%d: %s" % (n, t) for n, t in OBJETIVOS.items())
     + "\n\n") +
    "TAREFA: leia o documento e as metas candidatas. Atribua um ODS somente se o documento trata do que uma "
    "das metas candidatas descreve; indique essa meta e copie um trecho literal do documento que comprove. "
    "Use somente as metas candidatas. Se nenhuma se aplica, devolva a lista vazia.\n\n"
    "RESPONDA SO com JSON, sem texto fora dele, um item por ODS atribuido:\n" + _JSON)
# Variante para rotular em massa: so ODS e meta (gerar o trecho e o que mais custa).
# 30/10: a tarefa tambem deixa de pedir o trecho (antes so o JSON perdia o campo e o texto ainda pedia a citacao).
FIXO_CURTO = FIXO.replace(', "trecho": "ate 20 palavras copiadas do documento que comprovam"', "").replace(
    "indique essa meta e copie um trecho literal do documento que comprove. ", "indique essa meta. ")


def _fixo_v4(com_trecho):
    """Prompt v4 (D34): instrucao do rotulador do gabarito v7 (ferramentas/ods/avaliacao/gabarito_v7.py),
    para um documento e as metas candidatas. Mudou aqui? Mude la tambem (e vice-versa)."""
    trecho = (" e copie do documento, sem alterar nenhuma palavra e no idioma original, um trecho de até 20 "
              "palavras que mostre isso" if com_trecho else "")
    campo = ', "trecho": "<trecho copiado>"' if com_trecho else ""
    return (
        "<|im_start|>system\nVocê vai atribuir Objetivos de Desenvolvimento Sustentável (ODS) a um documento de um "
        "repositório. São os 17 ODS da Agenda 2030 da ONU e o ODS 18 do Brasil (igualdade étnico-racial).\n\n"
        "O documento pode ser de qualquer tipo e estar em qualquer idioma. O texto dele pode ter título, resumo, "
        "palavras-chave e um resumo gerado do texto completo, ou só parte disso.\n\n"
        "O único critério é o texto das metas candidatas dos ODS, anexadas depois do documento: as metas mais "
        "próximas do documento, cada uma com o texto da ONU, o texto do Brasil e os conceitos do Ipea (2026).\n\n"
        "OS 18 ODS (títulos oficiais):\n" + "\n".join("%d: %s" % (n, t) for n, t in OBJETIVOS.items()) + "\n\n"
        "- Baseie-se só no que está escrito no documento.\n"
        "- Atribua todos os ODS que se aplicam a ele: um, vários ou nenhum. Um ODS se aplica quando o documento "
        "trata do que uma das metas candidatas desse ODS descreve, no texto da ONU ou no do Brasil.\n"
        "- Para cada ODS atribuído, indique a meta que melhor se aplica" + trecho + ".\n"
        "- Antes de responder, confira cada ODS atribuído contra o texto da meta indicada.\n"
        "- Se nenhum ODS se aplica, devolva a lista vazia.\n\n"
        "Não use nenhum critério que não esteja escrito nas metas candidatas: nem o tipo, o formato ou a área do "
        "documento, nem o que você sabe sobre os ODS fora desse texto.\n\n"
        "Responda só com JSON, sem texto fora dele:\n"
        '{"ods": [{"n": <número do ODS>, "meta": "<código da meta>"' + campo + '}]}<|im_end|>\n')


if PROMPT_VERSAO == 4:
    FIXO, FIXO_CURTO = _fixo_v4(True), _fixo_v4(False)


def _fixo_v5(com_trecho):
    """Prompt v5 (29/10, a medir): base no v3 (o melhor medido; "somente se" mantem a precisao), sem presumir
    documento cientifico, sem exemplo com numero de ODS (a resposta e travada por JSON schema: so as metas
    candidatas, trecho antes da meta) e com a pergunta repetida no fim da mensagem (o Qwen le melhor a instrucao
    depois de um contexto longo). O ODS sai do codigo da meta."""
    return (
        "<|im_start|>system\nVoce atribui Objetivos de Desenvolvimento Sustentavel (ODS) a documentos de um "
        "repositorio institucional: artigos, teses, livros, relatorios, videos, boletins e outros, em qualquer "
        "idioma. Sao os 17 ODS da Agenda 2030 da ONU e o ODS 18 do Brasil (igualdade etnico-racial), com as metas "
        "globais da ONU e as metas brasileiras do Ipea (2026, Nacionalizacao da Agenda 2030).\n\n"
        "OS 18 ODS (titulos oficiais):\n" + "\n".join("%d: %s" % (n, t) for n, t in OBJETIVOS.items()) + "\n\n"
        "TAREFA: leia o documento e as metas candidatas. Atribua uma meta somente se o documento trata do que ela "
        "descreve. " + ("Para cada meta atribuida, copie primeiro um trecho literal do documento (palavra por "
                       "palavra, no idioma do documento, ate 20 palavras) que comprove, e depois o codigo da meta. "
                       if com_trecho else "") +
        "Use somente as metas candidatas e o que esta escrito no documento. Se nenhuma se aplica, devolva a lista "
        "vazia.\n\nResponda so com JSON no formato pedido.<|im_end|>\n")


def esquema_v5(candidatas, com_trecho=True):
    """JSON schema da resposta v5: lista de {trecho, meta}, meta so entre as candidatas (o servidor obriga)."""
    props = {"meta": {"type": "string", "enum": [c for c, _s in candidatas]}}
    req = ["meta"]
    if com_trecho:
        props = dict({"trecho": {"type": "string", "maxLength": 300}}, **props)
        req = ["trecho", "meta"]
    return {"type": "object", "properties": {"ods": {"type": "array", "items": {
        "type": "object", "properties": props, "required": req}}}, "required": ["ods"]}


if PROMPT_VERSAO == 5:
    FIXO, FIXO_CURTO = _fixo_v5(True), _fixo_v5(False)

# Prompt v6 (29/10, a medir; pedido da responsavel: "prompt completamente tecnico em JSON"): especificacao da tarefa
# em JSON e decisao independente sim/nao para CADA meta candidata (o schema exige uma chave por meta, na ordem),
# o que tira do texto qualquer indicacao de quantos ODS dar.
ESPEC_V6 = {
    "tarefa": "classificacao multirrotulo de um documento pelas metas dos Objetivos de Desenvolvimento Sustentavel (ODS)",
    "ods": {str(n): t for n, t in OBJETIVOS.items()},
    "entrada": {
        "documento": "texto de um item de repositorio institucional (qualquer tipo e idioma): titulo, resumo, "
                     "palavras-chave e/ou resumo gerado do texto completo, ou so parte disso",
        "metas_candidatas": "codigo da meta -> texto global (ONU), texto Brasil (Ipea 2026) e conceitos (Ipea 2026)"},
    "regras": [
        "avalie cada meta candidata de forma independente das outras",
        "aplica = true somente se o documento trata do que a meta descreve",
        "use apenas o que esta escrito no documento e o texto da meta",
        "se aplica = true: trecho = citacao literal do documento, palavra por palavra, no idioma do documento, "
        "ate 20 palavras; se aplica = false: trecho = \"\""],
    "saida": "JSON com uma chave por meta candidata: {\"aplica\": true ou false, \"trecho\": texto}"}


# 30/10 (teoria da responsavel: pedir trecho faz o modelo casar frases soltas com metas, em vez de julgar o documento
# inteiro): v6 sem trecho, so a decisao sim/nao por meta.
ESPEC_V6_SEM_TRECHO = dict(ESPEC_V6, regras=ESPEC_V6["regras"][:3],
                           saida="JSON com uma chave por meta candidata: {\"aplica\": true ou false}")


def _prompt_v6(texto_doc, candidatas, metas, com_trecho=True):
    def texto(cod):
        m = metas[cod]
        return MB.texto_metas(m) + (" | Conceitos (Ipea 2026): " + m["conceitos"] if m.get("conceitos") else "")
    entrada = {"documento": texto_doc, "metas_candidatas": {c: texto(c) for c, _s in candidatas}}
    return ("<|im_start|>system\n" + json.dumps(ESPEC_V6 if com_trecho else ESPEC_V6_SEM_TRECHO, ensure_ascii=False, indent=1) + "<|im_end|>\n"
            "<|im_start|>user\n" + json.dumps(entrada, ensure_ascii=False, indent=1) + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def esquema_v6(candidatas, com_trecho=True):
    item = {"type": "object", "properties": {"aplica": {"type": "boolean"}, "trecho": {"type": "string", "maxLength": 300}},
            "required": ["aplica", "trecho"]}
    if not com_trecho:
        item = {"type": "object", "properties": {"aplica": {"type": "boolean"}}, "required": ["aplica"]}
    cods = [c for c, _s in candidatas]
    return {"type": "object", "properties": {c: item for c in cods}, "required": cods}


def _ler_json_v6(txt):
    try:
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except (AttributeError, ValueError):
        return None
    out = ListaODS()
    out.tema = str(d.get("tema", ""))
    for cod, v in d.items():
        if isinstance(v, dict) and v.get("aplica") is True:
            n = int(str(cod).split(".")[0])
            if n not in [o["n"] for o in out]:
                out.append({"n": n, "meta": cod, "trecho": str(v.get("trecho", ""))})
    return out
# Experimento de 22/10 (variante D, pedido do responsavel): ODS_SO_LISTA=sim entrega so a lista dos 18 ODS, sem metas.
SO_LISTA = os.getenv("ODS_SO_LISTA", "nao") == "sim"
FIXO_SO_LISTA = (
    "<|im_start|>system\nVoce indexa producao cientifica pelos Objetivos de Desenvolvimento Sustentavel (ODS): "
    "metas globais da ONU e metas brasileiras do Ipea (2026, Nacionalizacao da Agenda 2030), incluindo o ODS 18.\n\n"
    "OS 18 ODS (titulos oficiais):\n" + "\n".join("%d: %s" % (n, t) for n, t in OBJETIVOS.items()) +
    "\n\nTAREFA: leia o documento. Atribua um ODS somente se o documento trata do que o objetivo descreve; copie um "
    "trecho literal do documento que comprove. Se nenhum se aplica, devolva a lista vazia.\n\n"
    "RESPONDA SO com JSON, sem texto fora dele, um item por ODS atribuido:\n"
    '{"ods": [{"n": 3, "trecho": "ate 20 palavras copiadas do documento que comprovam"}]}\n'
    'Lista vazia se nenhum ODS se aplica: {"ods": []}<|im_end|>\n')

# Prompt v2 (04-10/10/2026; rotulou o acervo): descricoes curtas dos ODS + metodo + regras por ODS + exemplos.
FIXO_V2 = (
    "<|im_start|>system\nVoce e bibliotecario(a) especialista em indexacao de producao cientifica pelos "
    "Objetivos de Desenvolvimento Sustentavel (ODS), seguindo as metas globais da ONU e as metas "
    "brasileiras do Ipea (2026, Nacionalizacao da Agenda 2030), incluindo o ODS 18 (igualdade etnico-racial).\n\n"
    "ODS:\n" + "\n".join("%d: %s" % (n, t) for n, t in OBJETIVOS_PROMPT.items()) +
    "\n\n" + CRITERIOS_PROMPT_V2 +
    "\nRESPONDA SO com JSON, sem texto fora dele, um item por ODS atribuido, com a meta que melhor se encaixa:\n" + _JSON)


def _caminho():
    if GGUF and os.path.exists(GGUF):
        return GGUF
    achados = glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                                     "models--" + REPO.replace("/", "--"), "snapshots", "*", ARQUIVO))
    if achados:
        return achados[0]
    # 02/11: modelos fora do SSD ficam no NAS (repo/arquivo). Os que cabem na RAM sao lidos uma vez ao subir o
    # servidor; os maiores que a RAM (Qwen3.8) sao lidos o tempo todo: esses vao para o SSD (ODS_PROFESSOR_GGUF)
    nas = os.path.join(NAS_LLM, *REPO.split("/"), *ARQUIVO.split("/"))
    if os.path.exists(nas):
        return nas
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, ARQUIVO)


def _llm():
    if "llm" not in _M:
        from llama_cpp import Llama
        _M["llm"] = Llama(model_path=_caminho(), n_gpu_layers=-1, n_ctx=CONTEXTO, verbose=False,
                          flash_attn=True, seed=0)
    return _M["llm"]


def _prompt(texto_doc, candidatas, metas, com_trecho=True):
    if SO_LISTA:                                    # variante D: so os 18 ODS, sem metas
        return (FIXO_SO_LISTA + "<|im_start|>user\nDOCUMENTO:\n" + texto_doc + "\n\nResponda so o JSON.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    linhas = []
    for cod, _s in candidatas:
        m = metas[cod]
        txt = "%s: %s" % (cod, MB.texto_metas(m))
        if m.get("conceitos"):                      # D27: conceitos importantes do Ipea 2026, inteiros
            txt += " | Conceitos (Ipea 2026): " + m["conceitos"]
        linhas.append(txt)
    return ((FIXO if com_trecho else FIXO_CURTO) + "<|im_start|>user\nDOCUMENTO:\n" + texto_doc +
            "\n\nMETAS CANDIDATAS (as %d mais proximas do documento" % len(candidatas) + " pelo significado; texto ONU, texto Brasil e conceitos do Ipea 2026):\n"
            + "\n".join(linhas) +
            ("\n\nQuais metas candidatas o documento trata? Atribua somente as que ele trata; se nenhuma, lista "
             "vazia. Responda so o JSON.<|im_end|>\n" if PROMPT_VERSAO == 5 else "\n\nResponda so o JSON.<|im_end|>\n")
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n")


class ListaODS(list):
    """Lista de ODS com o tema que o modelo escreveu (v8 com ODS_V8_TEMA)."""
    tema = ""


def _ler_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    out = ListaODS()
    out.tema = str(d.get("tema") or d.get("descricao") or "") if isinstance(d, dict) else ""
    for x in d.get("ods") or []:
        try:
            if isinstance(x, dict) and x.get("n") is None and x.get("meta"):   # v5: o ODS sai do codigo da meta
                n = int(str(x["meta"]).split(".")[0])
            else:
                n = int(x.get("n") if isinstance(x, dict) else x)
        except (TypeError, ValueError):
            continue
        if isinstance(x, dict) and x.get("atribuir") is False:     # v10b: o modelo justificou e decidiu nao atribuir
            out.descartados = getattr(out, "descartados", []) + [{"n": n, "meta": str(x.get("meta", "")),
                                                                   "justificativa": str(x.get("justificativa", ""))}]
            continue
        # v11: um item por meta (varias metas do mesmo ODS ficam) e lista de trechos, sem limite de palavras
        if 1 <= n <= 18 and (V8 == "v11" or n not in [o["n"] for o in out]):
            tr = x.get("trechos") if isinstance(x, dict) else None
            out.append({"n": n, "meta": str(x.get("meta", "")) if isinstance(x, dict) else "",
                        "trecho": " [...] ".join(str(y) for y in tr) if isinstance(tr, list) else
                        str(x.get("trecho", "")) if isinstance(x, dict) else ""})
            if isinstance(x, dict) and x.get("justificativa"):
                out[-1]["justificativa"] = str(x["justificativa"])
            if isinstance(x, dict) and x.get("fora_das_candidatas"):
                out[-1]["fora_das_candidatas"] = True
    return out


def carregar_cache():
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar_cache(cache):
    tmp = CACHE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, CACHE_JSON)


LLAMA_SERVER = os.getenv("ODS_LLAMA_SERVER", os.path.join(os.path.expanduser("~"), ".unsloth", "llama.cpp",
                                                          "build", "bin", "Release", "llama-server.exe"))


def iniciar_servidor(espera_s=int(os.getenv("ODS_LLM_ESPERA", "180")), aviso=print):   # Qwen3.8 no NAS: carga lenta
    """Sobe o llama-server (CUDA, 4 vagas em paralelo, so em 127.0.0.1, com chave) como processo
    independente. Devolve True se ficou de pe. Sem o executavel, o professor usa o modo lento."""
    import secrets
    import subprocess
    import time
    if servidor_ativo():
        return True
    if not os.path.exists(LLAMA_SERVER):
        aviso("llama-server nao encontrado (%s): professor no modo lento." % LLAMA_SERVER)
        return False
    if not os.path.exists(CHAVE_ARQ):
        with open(CHAVE_ARQ, "w") as f:
            f.write(secrets.token_hex(16))
    with open(CHAVE_ARQ) as f:
        chave = f.read().strip()
    amb = dict(os.environ)
    try:  # as DLLs do CUDA que o llama.cpp precisa vem com o PyTorch
        import torch
        amb["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + amb.get("PATH", "")
    except Exception:
        pass
    porta = SERVIDOR.rsplit(":", 1)[-1]
    log = open(os.path.join(config.LOGS, "llama_server.log"), "w")
    # sem -ngl: o llama.cpp ajusta sozinho o que cabe na GPU (forcar 99 transbordava para a memoria
    # compartilhada do Windows e ficava ~10x mais lento quando o contexto cresceu com os conceitos do Ipea)
    ngl = ["-ngl", _CFG["ngl"]] if _CFG.get("ngl") else []
    caminho = _caminho()
    # modelo no NAS: carrega inteiro na RAM ao subir (--load-mode none; este build nao tem --no-mmap), para nao reler pela rede durante a rodada
    nas = ["--load-mode", "none"] if caminho.replace("/", "\\").startswith("\\\\") else []
    subprocess.Popen([LLAMA_SERVER, "-m", caminho] + nas + ngl +["-c", str(CONTEXTO * PARALELO), "-np", str(PARALELO),
                      "-fa", "on", "--host", "127.0.0.1", "--port", porta, "--jinja", "--api-key", chave]
                     + ([] if _CFG.get("kwargs") or PENSAR else ["--reasoning-budget", "0"]) + _CFG["args"], env=amb, stdout=log, stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    t = time.time()
    while time.time() - t < espera_s:
        if servidor_ativo():
            aviso("Servidor do professor no ar (%s)." % SERVIDOR)
            return True
        time.sleep(3)
    aviso("Servidor do professor nao subiu em %ds (veja logs/llama_server.log)." % espera_s)
    return False


def parar_servidor():
    """Encerra o llama-server (libera a GPU)."""
    import subprocess
    subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)


def servidor_ativo():
    if not SERVIDOR:
        return False
    try:
        import requests
        return requests.get(SERVIDOR + "/health", timeout=3).status_code == 200
    except Exception:
        return False


def mensagens(prompt):
    """Prompt ChatML -> [system, user] (o texto nao muda; so o formato)."""
    sis = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.S).group(1)
    usr = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S).group(1)
    return [{"role": "system", "content": sis}, {"role": "user", "content": usr}]


def _completar(prompt, temperatura, n_predict=300, esquema=None):
    n_predict = max(n_predict, _CFG.get("n_predict", 0), 8000 if PENSAR else 0)
    if PENSAR:      # a gramatica do JSON valeria tambem para o raciocinio; com PENSAR o JSON e lido do texto final
        esquema = None
    if SERVIDOR and servidor_ativo():
        import requests
        cab = {}
        if os.path.exists(CHAVE_ARQ):
            with open(CHAVE_ARQ) as f:
                cab["Authorization"] = "Bearer " + f.read().strip()
        corpo = {"messages": mensagens(prompt), "max_tokens": n_predict, "temperature": temperatura, "seed": 0,
                 "chat_template_kwargs": dict({"enable_thinking": PENSAR}, **_CFG.get("kwargs", {})), "cache_prompt": True}
        if esquema:     # v5: o servidor so deixa sair JSON nesse formato (gramatica)
            corpo["response_format"] = {"type": "json_schema", "json_schema": {"name": "ods", "schema": esquema}}
        r = requests.post(SERVIDOR + "/v1/chat/completions", headers=cab, timeout=1800, json=corpo)
        if r.status_code != 200:     # inclui texto maior que o contexto: o servidor recusa, nunca corta
            raise RuntimeError("professor recusou (%s): %s" % (r.status_code, r.text[:300]))
        return r.json()["choices"][0]["message"].get("content") or ""
    r = _llm().create_completion(prompt, max_tokens=n_predict, temperature=temperatura, stop=["<|im_end|>"])
    return r["choices"][0]["text"]


# Prompt v8 (30/10, pedido da responsavel; em revisao, nada rodado): sem trecho, explica o que sao ODS e metas
# (codigo = numero do ODS + meta) e o formato das metas (META ONU / META IPEA / CONCEITOS IPEA). Duas formas:
# ODS_V8=texto (base v3, instrucao em portugues) e ODS_V8=tecnico (base v6, especificacao JSON, sim/nao por meta).
V8 = os.getenv("ODS_V8", "texto")      # texto | tecnico | nota (30/10: nota de 0 a 100 por meta; corte medido)
# Prompt v9 (02/11, responsavel: "muda o nome da versao do prompt para 9"): o v8 descricao + trecho (descricao do
# documento sem limite no lugar do tema de 30 palavras; trecho so como prova). ODS_PROMPT=9.
if PROMPT_VERSAO == 9:
    V8 = "descricao_trecho"
# Prompt v10 (03/11, em teste): justificativa antes da meta, sem tema e sem trecho, pode sair das candidatas (marcado)
if PROMPT_VERSAO == 10:
    V8 = os.getenv("ODS_V10_FORMA", "justificativa")   # justificativa | justificativa_decisao (v10b)
# Prompt v11 (04/11, responsavel): o v8 tema + trecho, deixando claro no comeco que o documento pode ter mais de uma meta,
# sem CONCEITOS IPEA (o professor recebe o mesmo que o gabarito) e sem segunda etapa. ODS_PROMPT=11.
if PROMPT_VERSAO == 11:
    V8 = "v11"
# Prompt v12 (06/11, responsavel): o v8 tema + trecho sem CONCEITOS; so muda o criterio, inspirado na regra geral dos juizes
# do OpenAlex (assunto principal ou contribuicao; mencao de passagem ou palavra em comum nao basta). ODS_PROMPT=12.
if PROMPT_VERSAO == 12:
    V8 = "tema_trecho"
# ODS_PENSAR=sim (30/10, pedido da responsavel): liga o raciocinio do modelo antes da resposta (Qwen thinking).
PENSAR = os.getenv("ODS_PENSAR", "nao") == "sim"
# ODS_SEM_CONCEITOS=sim (04/11, responsavel: o professor recebe o mesmo que o gabarito, so as metas): tira os CONCEITOS
# IPEA das metas candidatas e da explicacao do prompt v8/v9/v10.
SEM_CONCEITOS = os.getenv("ODS_SEM_CONCEITOS", "nao") == "sim" or PROMPT_VERSAO in (11, 12)


def _partes_meta(m):
    """Partes da meta para o v8: (rotulo, texto). Uma so parte quando o Ipea manteve o texto da ONU."""
    onu, br = m.get("onu") or "", m.get("brasil") or ""
    if onu and (m.get("mantida") or onu == br or not br):
        partes = [("meta_onu_e_ipea", onu)]
    else:
        partes = ([("meta_onu", onu)] if onu else []) + [("meta_ipea", br)]
    if m.get("conceitos") and not SEM_CONCEITOS:
        partes.append(("conceitos_ipea", m["conceitos"]))
    return partes


_ESTRUTURA_V8 = ("Os ODS são 18, numerados de 1 a 18. Cada ODS tem várias metas, numeradas a partir do número do "
                 "ODS: por exemplo, 3.1, 3.2 e 3.a são metas do ODS 3.")
FIXO_V8_TEXTO = (
    "<|im_start|>system\nVocê classifica documentos de um repositório institucional pelos Objetivos de "
    "Desenvolvimento Sustentável (ODS).\n\n" + _ESTRUTURA_V8 + "\n\nOS 18 ODS:\n" +
    "\n".join("%d: %s" % (n, t) for n, t in OBJETIVOS.items()) + "\n\n"
    "COMO AS METAS SÃO APRESENTADAS: cada meta candidata vem com o seu código e até três partes:\n"
    "- META ONU: o texto global da meta, da ONU.\n"
    "- META IPEA: o texto da mesma meta adaptado ao Brasil pelo Ipea (2026); pode mudar ou acrescentar temas ao "
    "texto da ONU.\n"
    "- CONCEITOS IPEA: a explicação, pelo Ipea, dos termos usados na meta.\n"
    "Quando o Ipea manteve o texto da ONU, aparece uma parte só, META ONU E IPEA. As metas que só existem no Brasil "
    "(por exemplo, as do ODS 18) não têm META ONU. Algumas metas não têm CONCEITOS IPEA. Em algumas metas, a META "
    "IPEA é dividida em submetas (por exemplo, 5.b.1 e 5.b.2br); a meta a indicar é sempre o código da meta (5.b).\n\n")
if SEM_CONCEITOS:
    FIXO_V8_TEXTO = (FIXO_V8_TEXTO.replace("até três partes", "até duas partes")
                     .replace("- CONCEITOS IPEA: a explicação, pelo Ipea, dos termos usados na meta.\n", "")
                     .replace(" Algumas metas não têm CONCEITOS IPEA.", ""))
# ODS_V8_TEMA=N (30/10): antes de decidir, o modelo escreve em ate N palavras do que o documento trata como um todo
# (tema) e, no v8 texto, uma justificativa de ate 25 palavras ANTES de cada meta (raciocinio antes da decisao). 0 = sem.
V8_TEMA = int(os.getenv("ODS_V8_TEMA", "0"))
V8_JUST = 25


def _fixo_v8_texto(tema):
    if V8 == "tema_trecho":
        # 30/10 (teste pedido pela responsavel): tema do documento inteiro primeiro; o trecho volta, mas so como
        # prova de que a meta aparece no documento, nao como ponto de partida da decisao.
        tarefa = ("TAREFA: leia o documento e as metas candidatas. O documento pode se relacionar com nenhum, um ou mais "
                  "ODS. Primeiro, escreva em até %d palavras do que o documento trata, considerando-o como um todo "
                  "(tema). Depois, a partir desse tema, atribua um ODS somente se o documento trata do que uma das "
                  "metas candidatas descreve, na META ONU ou na META IPEA; para cada ODS atribuído, indique essa meta "
                  "e copie do documento, sem alterar nenhuma palavra e no idioma original, um trecho de até 20 palavras "
                  "que comprove. O trecho é só a prova: a decisão vem do documento inteiro, não de uma frase isolada. "
                  "Use somente as metas candidatas. Se nenhuma se aplica, devolva a lista vazia.\n\n" % (tema or 30))
        if PROMPT_VERSAO == 12:
            tarefa = (tarefa.replace("somente se o documento trata", "somente se o assunto principal ou a contribuição do documento trata")
                      .replace("na META IPEA; para cada ODS atribuído", "na META IPEA. Uma menção de passagem ou uma palavra em comum não basta. Para cada ODS atribuído"))
        saida = ('{"tema": "<tema>", "ods": [{"meta": "3.3", "n": 3, "trecho": "<trecho copiado>"}]}\n'
                 'Lista vazia se nenhum ODS se aplica: {"tema": "<tema>", "ods": []}<|im_end|>\n')
    elif V8 == "v11":
        tarefa = ("TAREFA: um documento pode se relacionar com nenhuma, uma ou mais metas, de um ou mais ODS. As metas candidatas "
                  "são as mais próximas do documento pelo significado. "
                  "Leia o documento e as metas candidatas. Primeiro, escreva em até %d palavras do que o documento trata, "
                  "considerando-o como um todo (tema). Depois, a partir desse tema, analise cada meta candidata, lendo a META ONU e "
                  "a META IPEA, e atribua uma meta somente se o documento trata do que ela descreve. Para cada meta atribuída, primeiro copie do documento, sem alterar nenhuma palavra e no "
                  "idioma original, os trechos que justifiquem a atribuição; depois indique a meta. A decisão vem do documento "
                  "inteiro, não de uma frase isolada. Use somente as metas candidatas. Se nenhuma se aplica, devolva a lista vazia.\n\n" % (tema or 30))
        saida = ('{"tema": "<tema>", "ods": [{"trechos": ["<trecho copiado>", "<outro trecho, se houver>"], "meta": "3.3", "n": 3}]}\n'
                 'Lista vazia se nenhuma meta se aplica: {"tema": "<tema>", "ods": []}<|im_end|>\n')
    elif V8 == "descricao_trecho":
        # 02/11 (responsavel): o tema + trecho numa chamada so, mas com a DESCRICAO sem limite no lugar do tema de 30
        # palavras (mesmo texto da 1a chamada do duas_etapas)
        tarefa = ("TAREFA: leia o documento e as metas candidatas. O documento pode se relacionar com nenhum, um ou mais "
                  "ODS. Primeiro, descreva o documento como um todo, sem copiar frases: do que ele trata; o que estuda, "
                  "apresenta ou propõe; sobre quem ou onde, quando houver; e a que resultados ou conclusões chega, "
                  "quando houver (descrição). Depois, a partir dessa descrição, atribua um ODS somente se o documento "
                  "trata do que uma das metas candidatas descreve, na META ONU ou na META IPEA; para cada ODS atribuído, "
                  "indique essa meta e copie do documento, sem alterar nenhuma palavra e no idioma original, um trecho "
                  "de até 20 palavras que comprove. O trecho é só a prova: a decisão vem do documento inteiro, não de "
                  "uma frase isolada. Use somente as metas candidatas. Se nenhuma se aplica, devolva a lista vazia.\n\n")
        saida = ('{"descricao": "<descrição>", "ods": [{"meta": "3.3", "n": 3, "trecho": "<trecho copiado>"}]}\n'
                 'Lista vazia se nenhum ODS se aplica: {"descricao": "<descrição>", "ods": []}<|im_end|>\n')
    elif V8 == "justificativa":
        # 03/11 (prompt v10, pedido da responsavel): sem tema/descricao e sem trecho; para cada ODS, primeiro uma
        # justificativa com as palavras do modelo (antes da meta: raciocinio antes da decisao); pode sair das
        # candidatas, marcado. ODS_V10_LIMITE=sim limita a justificativa a uma ou duas frases.
        lim = " e em uma ou duas frases" if os.getenv("ODS_V10_LIMITE", "nao") == "sim" else ""
        tarefa = ("TAREFA: leia o documento e as metas candidatas. O documento pode se relacionar com nenhum, um ou mais "
                  "ODS. Analise cada meta candidata, lendo a META ONU e a META IPEA, e verifique se o documento, "
                  "considerado como um todo, trata do que ela descreve. Para cada ODS que você atribuir, escreva primeiro "
                  "a justificativa: explique, com as suas palavras%s, por que o documento trata do que a meta descreve, "
                  "considerando do que ele trata, o que estuda ou propõe, sobre quem ou onde e a que conclui, quando "
                  "houver. Depois indique a meta. Use preferencialmente as metas candidatas. Se o documento tratar "
                  "claramente de um ODS cujas metas não estão entre as candidatas, você pode atribuí-lo: indique o número "
                  "do ODS, escreva a justificativa e marque \"fora_das_candidatas\": true; nesse caso, indique a meta só "
                  "se tiver certeza do código. Se nenhum se aplica, devolva a lista vazia.\n\n" % lim)
        saida = ('{"ods": [{"justificativa": "<por que o documento trata do que a meta descreve>", "meta": "3.3", "n": 3, '
                 '"fora_das_candidatas": false}]}\nLista vazia se nenhum ODS se aplica: {"ods": []}<|im_end|>\n')
    elif V8 == "justificativa_decisao":
        # 03/11 (v10b, pedido da responsavel: mais precisao): sem "analise cada meta candidata"; para cada ODS que
        # considerar, justificativa e depois uma decisao explicita "atribuir": true/false (so true conta)
        tarefa = ("TAREFA: leia o documento e as metas candidatas. O documento pode se relacionar com nenhum, um ou mais "
                  "ODS. Para cada ODS que você considerar, escreva primeiro a justificativa: explique, com as suas "
                  "palavras, por que o documento trata ou não trata do que a meta descreve, considerando do que ele "
                  "trata, o que estuda ou propõe, sobre quem ou onde e a que conclui, quando houver. Depois decida: "
                  "\"atribuir\": true somente se a justificativa mostrar que o documento, considerado como um todo, trata "
                  "do que a meta descreve; \"atribuir\": false se o documento apenas menciona o tema ou tem com ele uma "
                  "relação indireta. Use preferencialmente as metas candidatas. Se o documento tratar claramente de um "
                  "ODS cujas metas não estão entre as candidatas, você pode considerá-lo: indique o número do ODS e "
                  "marque \"fora_das_candidatas\": true; nesse caso, indique a meta só se tiver certeza do código. Se "
                  "nenhum se aplica, devolva a lista vazia.\n\n")
        saida = ('{"ods": [{"justificativa": "<por que o documento trata ou não do que a meta descreve>", "meta": "3.3", '
                 '"n": 3, "atribuir": true, "fora_das_candidatas": false}]}\nLista vazia se nenhum ODS se aplica: '
                 '{"ods": []}<|im_end|>\n')
    elif V8 == "duas_etapas":
        # 02/11 (sugestao da revisao por agentes, aprovada pela responsavel): uma descricao do documento e escrita
        # numa chamada ANTES, sem ver as metas (FIXO_V8_SO_TEMA); aqui o modelo a recebe pronta e so escolhe as metas.
        # "Descricao" e nao "tema" (responsavel: um trabalho tem objeto, hipotese, conclusao... nao so um tema)
        tarefa = ("TAREFA: leia o documento, a descrição do documento e as metas candidatas. O documento pode se "
                  "relacionar com nenhum, um ou mais ODS. A DESCRIÇÃO DO DOCUMENTO foi escrita antes de ver as metas e "
                  "diz do que ele trata como um todo. A partir dessa descrição, atribua um ODS somente se o documento "
                  "trata do que uma das metas "
                  "candidatas descreve, na META ONU ou na META IPEA; para cada ODS atribuído, indique essa meta e copie "
                  "do documento, sem alterar nenhuma palavra e no idioma original, um trecho de até 20 palavras que "
                  "comprove. O trecho é só a prova: a decisão vem do documento inteiro, não de uma frase isolada. Use "
                  "somente as metas candidatas. Se nenhuma se aplica, devolva a lista vazia.\n\n")
        saida = ('{"ods": [{"meta": "3.3", "n": 3, "trecho": "<trecho copiado>"}]}\n'
                 'Lista vazia se nenhum ODS se aplica: {"ods": []}<|im_end|>\n')
    elif V8 == "tema_livre":
        # 30/10 (teste pedido pela responsavel): resumo do documento inteiro sem limite de tamanho, sem trecho.
        tarefa = ("TAREFA: leia o documento e as metas candidatas. O documento pode se relacionar com nenhum, um ou mais "
                  "ODS. Primeiro, resuma do que o documento trata, considerando-o como um todo (tema). Depois, a partir "
                  "desse tema, atribua um ODS somente se o documento trata do que uma das metas candidatas descreve, na "
                  "META ONU ou na META IPEA; indique essa meta. Use somente as metas candidatas. Se nenhuma se aplica, "
                  "devolva a lista vazia.\n\n")
        saida = ('{"tema": "<tema>", "ods": [{"meta": "3.3", "n": 3}]}\n'
                 'Lista vazia se nenhum ODS se aplica: {"tema": "<tema>", "ods": []}<|im_end|>\n')
    elif tema:
        tarefa = ("TAREFA: leia o documento e as metas candidatas. O documento pode se relacionar com nenhum, um ou mais "
                  "ODS. Primeiro, escreva em até %d palavras do que o documento trata, considerando-o como um todo "
                  "(tema). Depois, atribua um ODS somente se o documento trata do que uma das metas candidatas "
                  "descreve, na META ONU ou na META IPEA; para cada ODS atribuído, escreva em até %d palavras por que "
                  "o documento trata do que a meta descreve (justificativa) e indique essa meta. Use somente as metas "
                  "candidatas. Se nenhuma se aplica, devolva a lista vazia.\n\n" % (tema, V8_JUST))
        saida = ('{"tema": "<tema>", "ods": [{"justificativa": "<justificativa>", "meta": "3.3", "n": 3}]}\n'
                 'Lista vazia se nenhum ODS se aplica: {"tema": "<tema>", "ods": []}<|im_end|>\n')
    else:
        tarefa = ("TAREFA: leia o documento e as metas candidatas. O documento pode se relacionar com nenhum, um ou mais "
                  "ODS. Atribua um ODS somente se o documento trata do que uma das metas candidatas descreve, na META "
                  "ONU ou na META IPEA; indique essa meta. Use somente as metas candidatas. Se nenhuma se aplica, "
                  "devolva a lista vazia.\n\n")
        saida = '{"ods": [{"n": 3, "meta": "3.3"}]}\nLista vazia se nenhum ODS se aplica: {"ods": []}<|im_end|>\n'
    return FIXO_V8_TEXTO + tarefa + "RESPONDA SÓ com JSON, sem texto fora dele, um item por %s:\n" % ("meta atribuída" if V8 == "v11" else "ODS atribuído") + saida


_ROT_V8 = {"meta_onu": "META ONU", "meta_ipea": "META IPEA", "meta_onu_e_ipea": "META ONU E IPEA",
           "conceitos_ipea": "CONCEITOS IPEA"}


# ODS_V8=duas_etapas, 1a chamada: so o documento, sem as metas
# 02/11 (responsavel): sem limite de palavras; nao presume documento cientifico (D34): "quando houver"
FIXO_V8_SO_TEMA = ("<|im_start|>system\nVocê lê documentos de um repositório institucional.\n\nTAREFA: descreva o "
                   "documento como um todo, sem copiar frases: do que ele trata; o que estuda, apresenta ou propõe; sobre "
                   "quem ou onde, quando houver; e a que resultados ou conclusões chega, quando houver. Responda só com "
                   "a descrição.<|im_end|>\n")


def _prompt_v8_so_tema(texto_doc):
    return (FIXO_V8_SO_TEMA + "<|im_start|>user\nDOCUMENTO:\n" + texto_doc +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _prompt_v8_texto(texto_doc, candidatas, metas, tema=""):
    blocos = ["[%s]\n" % c + "\n".join("  %s: %s" % (_ROT_V8[r], t) for r, t in _partes_meta(metas[c]))
              for c, _s in candidatas]
    if tema:        # duas_etapas: a descricao da 1a chamada vem depois do documento
        texto_doc = texto_doc + "\n\nDESCRIÇÃO DO DOCUMENTO (escrita antes de ver as metas):\n" + tema
    return (_fixo_v8_texto(V8_TEMA) + "<|im_start|>user\nDOCUMENTO:\n" + texto_doc +
            "\n\nMETAS CANDIDATAS (as %d metas mais próximas do documento pelo significado):\n" % len(candidatas) +
            "\n\n".join(blocos) + "\n\nResponda só o JSON.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


ESPEC_V8 = {
    "tarefa": "classificação multirrótulo de um documento pelas metas dos Objetivos de Desenvolvimento Sustentável (ODS)",
    "estrutura_dos_ods": _ESTRUTURA_V8,
    "ods": {str(n): t for n, t in OBJETIVOS.items()},
    "entrada": {
        "documento": "texto de um item de repositório institucional (qualquer tipo e idioma): título, resumo, "
                     "palavras-chave e/ou resumo gerado do texto completo, ou só parte disso",
        "metas_candidatas": {
            "chave": "código da meta",
            "meta_onu": "texto global da meta, da ONU; ausente nas metas que só existem no Brasil (ex.: ODS 18)",
            "meta_ipea": "texto da mesma meta adaptado ao Brasil pelo Ipea (2026); pode mudar ou acrescentar temas; "
                         "pode vir dividido em submetas (ex.: 5.b.1, 5.b.2br), que pertencem à meta da chave",
            "meta_onu_e_ipea": "no lugar de meta_onu e meta_ipea, quando o Ipea manteve o texto da ONU",
            "conceitos_ipea": "explicação, pelo Ipea, dos termos usados na meta; pode faltar"}},
    "regras": [
        "o documento pode se relacionar com nenhum, um ou mais ODS",
        "avalie cada meta candidata de forma independente das outras",
        "aplica = true somente se o documento trata do que a meta descreve, na meta_onu ou na meta_ipea",
        "use apenas o que está escrito no documento e o texto da meta"],
    "saida": "JSON com uma chave por meta candidata: {\"aplica\": true ou false}"}


def espec_v8(tema):
    if not tema:
        return ESPEC_V8
    return dict(ESPEC_V8, regras=["primeiro, em tema, escreva em até %d palavras do que o documento trata, "
                                  "considerando-o como um todo" % tema] + ESPEC_V8["regras"],
                saida="JSON: primeiro a chave tema (texto); depois uma chave por meta candidata: "
                      "{\"aplica\": true ou false}")


def esquema_v8_tecnico(candidatas, tema):
    e = esquema_v6(candidatas, False)
    if tema:     # tema primeiro: o servidor obriga a ordem das chaves
        e = {"type": "object", "properties": dict({"tema": {"type": "string", "maxLength": 12 * tema}}, **e["properties"]),
             "required": ["tema"] + e["required"]}
    return e


def _prompt_v8_tecnico(texto_doc, candidatas, metas):
    entrada = {"documento": texto_doc, "metas_candidatas": {c: dict(_partes_meta(metas[c])) for c, _s in candidatas}}
    return ("<|im_start|>system\n" + json.dumps(espec_v8(V8_TEMA), ensure_ascii=False, indent=1) + "<|im_end|>\n"
            "<|im_start|>user\n" + json.dumps(entrada, ensure_ascii=False, indent=1) + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


ESPEC_V8_NOTA = dict(ESPEC_V8, regras=[
    "o documento pode se relacionar com nenhum, um ou mais ODS",
    "avalie cada meta candidata de forma independente das outras",
    "nota = número inteiro de 0 a 100: o quanto o documento, considerado como um todo, trata do que a meta descreve, "
    "na meta_onu ou na meta_ipea (0 = não trata; 100 = trata plenamente)",
    "use apenas o que está escrito no documento e o texto da meta"],
    saida="JSON com uma chave por meta candidata: {\"nota\": inteiro de 0 a 100}")


def _prompt_v8_nota(texto_doc, candidatas, metas):
    entrada = {"documento": texto_doc, "metas_candidatas": {c: dict(_partes_meta(metas[c])) for c, _s in candidatas}}
    return ("<|im_start|>system\n" + json.dumps(ESPEC_V8_NOTA, ensure_ascii=False, indent=1) + "<|im_end|>\n"
            "<|im_start|>user\n" + json.dumps(entrada, ensure_ascii=False, indent=1) + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def esquema_v8_nota(candidatas):
    item = {"type": "object", "properties": {"nota": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["nota"]}
    cods = [c for c, _s in candidatas]
    return {"type": "object", "properties": {c: item for c in cods}, "required": cods}


def _ler_json_nota(txt):
    """Todas as metas com a nota (o corte que vira ODS e medido depois, contra o gabarito)."""
    try:
        d = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except (AttributeError, ValueError):
        return None
    out = ListaODS()
    for cod, v in d.items():
        try:
            out.append({"n": int(str(cod).split(".")[0]), "meta": cod, "nota": int(v["nota"] if isinstance(v, dict) else v)})
        except (TypeError, ValueError, KeyError):
            continue
    return out


def rotular(texto_doc, candidatas, metas, tentativas=2, com_trecho=True):
    """Lista de ODS do professor para um documento (None se a resposta nao for JSON valido)."""
    if PROMPT_VERSAO in (8, 9, 10, 11, 12) and V8 == "nota":
        p, esq, ler = _prompt_v8_nota(texto_doc, candidatas, metas), esquema_v8_nota(candidatas), _ler_json_nota
    elif PROMPT_VERSAO in (8, 9, 10, 11, 12) and V8 == "tecnico":
        p, esq, ler = (_prompt_v8_tecnico(texto_doc, candidatas, metas), esquema_v8_tecnico(candidatas, V8_TEMA),
                       _ler_json_v6)
    elif PROMPT_VERSAO in (8, 9, 10, 11, 12) and V8 == "duas_etapas":
        tema = _completar(_prompt_v8_so_tema(texto_doc), 0, n_predict=1500).strip()   # sem limite de palavras

        def ler(txt, tema=tema):
            out = _ler_json(txt)
            if out is not None:
                out.tema = tema
            return out
        p, esq = _prompt_v8_texto(texto_doc, candidatas, metas, tema), None
    elif PROMPT_VERSAO in (8, 9, 10, 11, 12):
        p, esq, ler = _prompt_v8_texto(texto_doc, candidatas, metas), None, _ler_json
    elif PROMPT_VERSAO == 6:
        p, esq, ler = (_prompt_v6(texto_doc, candidatas, metas, com_trecho), esquema_v6(candidatas, com_trecho),
                       _ler_json_v6)
    else:
        p, ler = _prompt(texto_doc, candidatas, metas, com_trecho), _ler_json
        esq = esquema_v5(candidatas, com_trecho) if PROMPT_VERSAO == 5 else None
    for t in range(tentativas):
        out = ler(_completar(p, 0 if t == 0 else 0.3, n_predict=3000 if PROMPT_VERSAO in (8, 9, 10, 11, 12) and V8 in ("tema_livre", "v11", "justificativa", "justificativa_decisao") else 1500 if PROMPT_VERSAO == 6 or (PROMPT_VERSAO in (8, 9, 10, 11, 12) and (V8 in ("tecnico", "nota", "tema_trecho", "v11", "duas_etapas", "descricao_trecho", "justificativa", "justificativa_decisao") or V8_TEMA)) else 300, esquema=esq))
        if out is not None:
            return out
    return None


def rotular_lote(pares, metas, com_trecho=False):
    """pares: [(texto_doc, candidatas)] -> lista de resultados, em paralelo no servidor."""
    n = PARALELO if servidor_ativo() else 1
    prog = os.getenv("ODS_PROGRESSO")          # arquivo "feitos total" para o monitor de tarefas longas
    # 03/11 (o PC reiniciou no meio de uma rodada de 2h30 e tudo se perdeu): ODS_CHECKPOINT=arquivo.jsonl grava cada
    # resposta assim que sai e, ao recomecar, reaproveita as ja feitas (chave: modelo, prompt, documento e candidatas)
    ck = os.getenv("ODS_CHECKPOINT")
    feito = {}
    if ck and os.path.exists(ck):
        for ln in open(ck, encoding="utf-8"):
            try:
                x = json.loads(ln)
                feito[x["k"]] = (x["r"], x.get("tema", ""))
            except ValueError:
                continue          # linha cortada por queda no meio da gravacao
    feitos = [0]
    trava = threading.Lock()

    def chave(pc):
        import hashlib
        base = json.dumps([NOME, PROMPT_VERSAO, V8, V8_TEMA, com_trecho, pc[0], [c for c, _s in pc[1]]], ensure_ascii=False)
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def um(pc):
        k = chave(pc) if ck else None
        if k in feito:
            r, tema = feito[k]
            if r is not None:
                r = ListaODS(r)
                r.tema = tema
        else:
            r = rotular(pc[0], pc[1], metas, com_trecho=com_trecho)
            if ck:
                with trava:
                    with open(ck, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": k, "r": r, "tema": getattr(r, "tema", "")}, ensure_ascii=False) + "\n")
        if prog:
            with trava:
                feitos[0] += 1
                open(prog, "w").write("%d %d" % (feitos[0], len(pares)))
        return r
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(um, pares))
