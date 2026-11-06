# -*- coding: utf-8 -*-
"""Segunda opiniao do LLM local na faixa de revisao do DeCS (decisao do responsavel, 05/10; a medir).

O classificador do DeCS aprende com os catalogadores (o professor continua sendo a indexacao humana). Os
descritores com probabilidade entre o limiar de revisao e o de gravacao automatica hoje vao para revisao
humana. Aqui o LLM local (o mesmo professor do ODS, arca/ods_professor) le o documento INTEIRO e os
candidatos dessa faixa (nome + nota de escopo + sinonimos) e diz quais o documento de fato trata.
So vira parte do fluxo (config.DECS_SEGUNDA_OPINIAO) depois de medido em
ferramentas/decs/avaliar_segunda_opiniao_decs.py contra os catalogadores.
"""
import json
import re

from . import decs_acervo
from . import ods_professor as P

INSTRUCOES = (
    "Voce e bibliotecario(a) indexador(a) de ciencias da saude, especialista no vocabulario DeCS/MeSH. "
    "Recebe um documento e alguns descritores CANDIDATOS (com nota de escopo e sinonimos). Confirme so os "
    "descritores que representam um ASSUNTO TRATADO no documento (objeto, populacao, metodo, doenca, "
    "intervencao ou resultado), como um catalogador faria. Descarte o que aparece so de passagem, o que e "
    "generico demais para o documento e o que nao corresponde ao sentido da nota de escopo. Na duvida, "
    "descarte.\nRESPONDA SO com JSON: {\"confirmar\": [\"UI1\", \"UI2\"]} (lista vazia se nenhum).")


def _prompt(item, candidatos, vocab):
    texto = decs_acervo.texto_do_item(item, None)           # documento inteiro
    linhas = ["%s: %s" % (ui, vocab.texto_descritor(ui)) for ui, _p in candidatos]
    return ("<|im_start|>system\n" + INSTRUCOES + "<|im_end|>\n<|im_start|>user\nDOCUMENTO:\n" + texto +
            "\n\nDESCRITORES CANDIDATOS:\n" + "\n".join(linhas) +
            "\n\nQuais candidatos o documento trata? Responda so o JSON.<|im_end|>\n<|im_start|>assistant\n")


def confirmar(item, candidatos, vocab, tentativas=2):
    """candidatos: [(ui, probabilidade)] da faixa de revisao -> lista de UIs confirmados (None se falhar)."""
    if not candidatos:
        return []
    validos = {ui for ui, _p in candidatos}
    p = _prompt(item, candidatos, vocab)
    for t in range(tentativas):
        m = re.search(r"\{.*\}", P._completar(p, 0 if t == 0 else 0.3, n_predict=200), re.S)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        return [ui for ui in d.get("confirmar") or [] if ui in validos]
    return None
