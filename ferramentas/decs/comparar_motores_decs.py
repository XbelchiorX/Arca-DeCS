# -*- coding: utf-8 -*-
"""Compara os dois motores de DeCS (decisao do responsavel, 2026-10-16), os dois com o Qwen3-Embedding-8B,
nos MESMOS itens: os ultimos N depositados que ja tem DeCS do catalogador (treinar_avaliar.selecionar_amostra).

  A) motor de embedding: o classificador treinado com os catalogadores (o que treinar_avaliar --amostra acabou
     de treinar); galho pelo modelo de galho;
  B) motor de IA: o Qwen3.5-9B (cabe inteiro na GPU) le o registro INTEIRO e escolhe entre os candidatos que o
     embedding traz (os 30 melhores do classificador + os 30 descritores mais parecidos pelo texto), JA
     escolhendo o caminho na arvore (so caminhos que existem no servidor) e citando um trecho literal.
     Filtros automaticos: trecho que nao existe no texto e trecho repetido em mais de 2 descritores saem.

Mede contra o catalogador: descritores (precisao/recall/F1), galho (onde o catalogador gravou caminho) e o que
cada motor sugere de NOVO (o catalogador nao pos). Gera planilha de revisao com 30 itens para o responsavel.
Saidas: saidas/avaliacao/comparacao_motores_decs.{md,json} e saidas/avaliacao/revisao_motores_decs_30.xlsx.

Uso (depois de treinar_avaliar.py --amostra com DECS_MODELO_EMBEDDING=qwen3-8b, com os modelos ainda no lugar):
  set DECS_MODELO_EMBEDDING=qwen3-8b && python ferramentas/decs/comparar_motores_decs.py --teste 500
"""
import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("ODS_PROFESSOR", "qwen3.5-9b")
os.environ.setdefault("ODS_LLM_PARALELO", "2")
os.environ.setdefault("ODS_LLM_CONTEXTO", "16384")
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "ferramentas", "decs"))
import torch  # noqa: E402
from arca import config, decs_acervo, decs_galho, decs_indexador as IX, decs_vocabulario as V  # noqa: E402
from arca import ods_professor as P  # noqa: E402
from treinar_avaliar import duplicatas, selecionar_amostra  # noqa: E402

N_CLASSIF = 30
N_DENSO = 30

INSTR = (
    "Voce e bibliotecario(a) indexador(a) de ciencias da saude, especialista no DeCS. Recebe UM documento e uma "
    "lista de descritores candidatos do DeCS; cada candidato tem nome, nota de escopo, sinonimos e os CAMINHOS "
    "possiveis na arvore (numerados). Julgue os candidatos UM A UM e escolha so os que representam um ASSUNTO "
    "PRINCIPAL do documento (objeto, populacao, local, metodo, doenca, intervencao ou resultado). Nao escolha o que "
    "aparece so de passagem, o que e generico demais para o documento ou o que contraria a nota de escopo. Na "
    "duvida, nao escolha. Para cada escolhido: indique o NUMERO do caminho que melhor corresponde ao sentido no "
    "documento e copie, do documento, um trecho LITERAL curto (ate 25 palavras) que comprove ESSE descritor; "
    "trechos diferentes para descritores diferentes.\n"
    "RESPONDA SO com JSON: {\"descritores\": [{\"ui\": \"D000000\", \"caminho\": 1, \"trecho\": \"...\"}]}")


def _norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip().lower()


def _metricas(gold, pred):
    tp = fp = fn = 0
    for h in gold:
        g, p = set(gold[h]), set(pred.get(h, ()))
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precisao": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3),
            "por_item": round((tp + fp) / max(len(gold), 1), 2), "acertos": tp, "sugeridos": tp + fp}


def _prompt(texto, cands, vocab):
    linhas = []
    for u in cands:
        cams = vocab.caminhos(u)
        linhas.append("%s: %s\n   caminhos: %s" % (u, vocab.texto_descritor(u),
                      " | ".join("%d) %s" % (i + 1, c) for i, c in enumerate(cams))))
    return ("<|im_start|>system\n" + INSTR + "<|im_end|>\n<|im_start|>user\nDOCUMENTO:\n" + texto +
            "\n\nDESCRITORES CANDIDATOS:\n" + "\n".join(linhas) +
            "\n\nResponda so o JSON.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _ler(resp):
    m = re.search(r"\{.*\}", resp or "", re.S)
    if m:
        try:
            return json.loads(m.group(0)).get("descritores") or []
        except ValueError:
            pass
    out = []
    for x in re.findall(r"\{[^{}]*\"ui\"[^{}]*\}", resp or ""):
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teste", type=int, default=500)
    ap.add_argument("--treino", type=int, default=4000)
    ap.add_argument("--memoria", type=int, default=8000)
    args = ap.parse_args()
    t0 = time.time()
    vocab = V.carregar_vocabulario(permitir_arvore_bireme=True)
    acervo, ordem = selecionar_amostra(decs_acervo.carregar_acervo(), vocab, args.teste, args.treino, args.memoria)
    teste = ordem[:args.teste]
    ix = IX.montar_indexador(vocab, acervo, com_modelo=True)
    if ix.modelo is None:
        sys.exit("Modelo do motor A nao encontrado: rode antes treinar_avaliar.py --amostra (qwen3-8b).")
    ix.recalcular_prior(excluir=teste)
    galho = decs_galho.EscolhaDeGalho(vocab, acervo, vetores=ix.V, handles=ix.handles)
    lim_auto, lim_rev = ix.limiar["automatico"], ix.limiar["revisao"]
    print("pronto (%.0fs): %d itens de teste; limiares A: automatico %.2f, revisao %.2f"
          % (time.time() - t0, len(teste), lim_auto, lim_rev), flush=True)

    # 1) motor A e candidatos do B (tudo que usa a GPU do indice), antes de subir o LLM
    gold, A_auto, A_rev, cands, textos, vets = {}, {}, {}, {}, {}, {}
    for h in teste:
        vet = ix.V[ix.pos[h]]
        vets[h] = vet
        pts = ix.pontuar(acervo[h], vet, excluir=duplicatas(ix, h, vet), proprio=h)
        gold[h] = list(ix.gabarito.get(h, []))
        A_auto[h] = [u for u, p in pts if p >= lim_auto]
        A_rev[h] = [u for u, p in pts if p >= lim_rev]
        sims = (ix.Dg.float() @ torch.as_tensor(vet, dtype=torch.float32, device=ix.dev)).cpu().numpy()
        densos = [ix.uis_descr[i] for i in sims.argsort()[::-1][:N_DENSO]]
        cands[h] = [u for u in dict.fromkeys([u for u, _p in pts[:N_CLASSIF]] + densos) if vocab.caminhos(u)]
        textos[h] = decs_acervo.texto_do_item(acervo[h], None)
    del ix.Vviz, ix.Dg
    torch.cuda.empty_cache()

    # 2) motor B
    if not P.iniciar_servidor(espera_s=600):
        sys.exit("LLM nao subiu")
    t1 = time.time()

    def chamar(h):
        try:
            return P._completar(_prompt(textos[h], cands[h], vocab), 0, n_predict=1500)
        except Exception as e:                      # um item com problema nao derruba os outros
            print("  %s: LLM falhou (%s)" % (h, str(e)[:120]), flush=True)
            return ""

    with ThreadPoolExecutor(P.PARALELO) as ex:
        resps = dict(zip(teste, ex.map(chamar, teste)))
    sem_resposta = sum(1 for h in teste if not resps[h])
    s_item = (time.time() - t1) / len(teste)
    P.parar_servidor()
    B, B_cam, B_det, descartes = {}, {}, {}, {"fora_da_lista": 0, "trecho_inexistente": 0, "trecho_repetido": 0}
    for h in teste:
        usos, sel, cams, det = {}, [], {}, []
        for e in _ler(resps[h]):
            u, tr = e.get("ui"), (e.get("trecho") or "").strip()
            if u not in cands[h]:
                descartes["fora_da_lista"] += 1
                continue
            if not tr or _norm(tr) not in _norm(textos[h]):
                descartes["trecho_inexistente"] += 1
                continue
            usos[_norm(tr)] = usos.get(_norm(tr), 0) + 1
            if usos[_norm(tr)] > 2:
                descartes["trecho_repetido"] += 1
                continue
            caminhos = vocab.caminhos(u)
            try:
                c = caminhos[int(e.get("caminho", 1)) - 1]
            except (ValueError, IndexError, TypeError):
                c = caminhos[0]
            sel.append(u)
            cams[u] = c
            det.append({"ui": u, "caminho": c, "trecho": tr})
        B[h], B_cam[h], B_det[h] = list(dict.fromkeys(sel)), cams, det

    # 3) galho: onde o catalogador gravou caminho (multigalho), quem acerta?
    gal = {"A_modelo_de_galho": [0, 0], "B_llm": [0, 0], "B_com_modelo_de_galho": [0, 0]}
    for h in teste:
        humano = galho.humano_por_item.get(h, {})
        for nome, uis, cam_de in (("A_modelo_de_galho", A_rev[h], None), ("B_llm", B[h], B_cam[h]),
                                  ("B_com_modelo_de_galho", B[h], None)):
            ctxg = galho.contexto(h, [vocab.nome(u) for u in uis], vets[h])
            for u in uis:
                if u in humano and len(vocab.caminhos(u)) > 1:
                    c = cam_de[u] if cam_de else galho.escolher(u, excluir_handle=h, contexto=ctxg)[0]
                    gal[nome][0] += c == humano[u]
                    gal[nome][1] += 1
    gal = {k: {"acertos": a, "casos": n, "taxa": round(a / n, 3) if n else None} for k, (a, n) in gal.items()}

    res = {"itens": len(teste), "embedding": config.MODELO_EMBEDDING, "llm": P.NOME,
           "A_gravacao_automatica": _metricas(gold, A_auto), "A_ate_revisao": _metricas(gold, A_rev),
           "B_llm": _metricas(gold, B), "galho_multiplos_caminhos": gal, "B_descartes": descartes,
           "B_sem_resposta": sem_resposta,
           "B_segundos_por_item": round(s_item, 1),
           "B_dias_acervo_inteiro": round(s_item * 69500 / 86400, 1),
           "novos_A_por_item": round(sum(len(set(A_rev[h]) - set(gold[h])) for h in teste) / len(teste), 2),
           "novos_B_por_item": round(sum(len(set(B[h]) - set(gold[h])) for h in teste) / len(teste), 2)}
    out = config.SAIDAS_AVALIACAO
    with open(os.path.join(out, "comparacao_motores_decs.json"), "w", encoding="utf-8") as f:
        json.dump({"resumo": res, "itens": {h: {"humano": gold[h], "A": A_rev[h], "B": B_det[h], "candidatos": cands[h],
                                                "resposta_llm": resps[h]} for h in teste}}, f, ensure_ascii=False, indent=1)
    linhas = ["# Motores de DeCS: embedding (A) x IA (B)", "",
              "%d itens: os ultimos depositados que ja tem DeCS do catalogador. Embedding: %s. LLM: %s." %
              (len(teste), config.MODELO_EMBEDDING, P.NOME), "",
              "| Motor | Precisao | Recall | F1 | Descritores por item |", "|---|---|---|---|---|"]
    for nome, k in (("A - grava sozinho (>= %.2f)" % lim_auto, "A_gravacao_automatica"),
                    ("A - inclui revisao (>= %.2f)" % lim_rev, "A_ate_revisao"), ("B - LLM", "B_llm")):
        m = res[k]
        linhas.append("| %s | %.1f%% | %.1f%% | %.3f | %.1f |" % (nome, 100 * m["precisao"], 100 * m["recall"],
                                                                 m["f1"], m["por_item"]))
    linhas += ["", "Galho (so descritores com mais de um caminho, onde o catalogador gravou o caminho):", ""]
    for k, v in gal.items():
        linhas.append("- %s: %s de %s (%s)" % (k, v["acertos"], v["casos"], v["taxa"]))
    linhas += ["", "Sugestoes que o catalogador nao pos, por item: A %.2f | B %.2f (julgar na planilha de revisao)"
               % (res["novos_A_por_item"], res["novos_B_por_item"]),
               "B descartados pelos filtros: %s" % descartes,
               "B: %.1f s por item (acervo inteiro: ~%.1f dias nesta GPU)" % (s_item, res["B_dias_acervo_inteiro"])]
    with open(os.path.join(out, "comparacao_motores_decs.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print("\n".join(linhas), flush=True)

    # 4) planilha de revisao (30 itens sorteados entre os de teste)
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    amostra = random.Random(20261016).sample(teste, min(30, len(teste)))
    wb = Workbook()
    ws = wb.active
    ws.title = "Revisao"
    ws.append(["Link ARCA", "Texto (inteiro)", "DeCS do catalogador", "Motor A (embedding): descritor e caminho",
               "Motor B (IA): descritor, caminho e trecho", "Qual motor indexou melhor? (A/B/empate)", "Comentario"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for h in amostra:
        ctxg = galho.contexto(h, [vocab.nome(u) for u in A_rev[h]], vets[h])
        a = "\n".join("%s%s -> %s" % ("" if u in gold[h] else "[NOVO] ", vocab.nome(u),
                                        galho.escolher(u, excluir_handle=h, contexto=ctxg)[0]) for u in A_rev[h])
        b = "\n".join("%s%s -> %s\n   \"%s\"" % ("" if d["ui"] in gold[h] else "[NOVO] ", vocab.nome(d["ui"]),
                                                   d["caminho"], d["trecho"]) for d in B_det[h])
        ws.append(["https://arca.fiocruz.br/handle/" + h, textos[h], "\n".join(vocab.nome(u) for u in gold[h]), a, b, "", ""])
    for col, w in zip("ABCDEFG", (24, 80, 30, 60, 70, 18, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(os.path.join(out, "revisao_motores_decs_30.xlsx"))
    print("planilha: %s (%.1f h no total)" % (os.path.join(out, "revisao_motores_decs_30.xlsx"), (time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
