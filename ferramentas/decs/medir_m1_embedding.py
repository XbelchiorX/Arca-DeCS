# -*- coding: utf-8 -*-
"""Comparacao dos motores de DeCS com o Qwen3-Embedding-8B (decisao do responsavel, 2026-10-16), sem mexer no
que esta em uso. Titulo, palavras-chave e resumo vao INTEIROS (o 8B le texto longo).

0. coleta o acervo de hoje (D4, so leitura);
1. guarda uma copia dos modelos atuais (e5) em DADOS_LOCAIS/modelos_e5-large/ e do relatorio atual;
2. MOTOR A: treinar_avaliar.py --amostra --teste 500 com DECS_MODELO_EMBEDDING=qwen3-8b (teste = os ultimos 500
   depositados com DeCS do catalogador; treino 4.000 e memoria 8.000 sorteados entre os mais antigos);
3. MOTOR B: comparar_motores_decs.py --teste 500 (Qwen3.5-9B nos mesmos 500 itens) e a planilha de revisao;
4. move os modelos do 8B para DADOS_LOCAIS/modelos_qwen3-8b/ e devolve os e5 ao lugar.
Registro: logs/decs_m1.log. Se parar no meio, rodar de novo continua (os vetores ficam em cache).
Uso: python ferramentas/decs/medir_m1_embedding.py
"""
import os
import shutil
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
from arca import config  # noqa: E402

ARQS = ["modelo_decs.pkl", "modelo_galho.pkl", "rede_decs.pt"]
E5 = os.path.join(config.DADOS_LOCAIS, "modelos_e5-large")
Q8 = os.path.join(config.DADOS_LOCAIS, "modelos_qwen3-8b")
AV = config.SAIDAS_AVALIACAO
LOG = os.path.join(config.LOGS, "decs_m1.log")
TESTE = "500"


def log(msg):
    linha = time.strftime("%d/%m %H:%M:%S ") + msg
    print(linha, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def copiar(nomes, de, para, mover=False):
    os.makedirs(para, exist_ok=True)
    for n in nomes:
        o = os.path.join(de, n)
        if os.path.exists(o):
            (shutil.move if mover else shutil.copy2)(o, os.path.join(para, n))


def rodar(script, args, amb):
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-u", os.path.join(RAIZ, "ferramentas", "decs", script)] + args,
                           env=amb, stdout=f, stderr=subprocess.STDOUT, cwd=RAIZ)
    log("%s terminou (codigo %d) em %.1f h" % (script, r.returncode, (time.time() - t0) / 3600))
    return r.returncode == 0


def main():
    log("=== Motores de DeCS com Qwen3-Embedding-8B (A embedding x B IA), %s registros ===" % TESTE)
    from arca import dspace, decs_coleta
    idade_h = (time.time() - os.path.getmtime(config.ACERVO_JSON)) / 3600 if os.path.exists(config.ACERVO_JSON) else 99
    if idade_h < 24 and "--coletar" not in sys.argv:
        # retomada: a mesma foto do acervo mantem os mesmos 500 itens de teste e reaproveita os vetores
        log("acervo coletado ha %.0f h: usando a mesma copia (para coletar de novo: --coletar)" % idade_h)
    else:
        t0 = time.time()
        r = decs_coleta.coletar(dspace.conectar(), progresso=log)
        log("acervo coletado hoje (%.0f min): %s" % ((time.time() - t0) / 60, r))
    copiar(ARQS, config.DADOS_LOCAIS, E5)
    copiar(["avaliacao.md", "avaliacao.json"], AV, os.path.join(AV, "e5-large_atual"))
    log("modelos e5 guardados em %s" % E5)
    amb = dict(os.environ, DECS_MODELO_EMBEDDING="qwen3-8b", PYTHONIOENCODING="utf-8")
    try:
        if rodar("treinar_avaliar.py", ["--amostra", "--teste", TESTE], amb):
            for n in ("avaliacao.md", "avaliacao.json"):
                if os.path.exists(os.path.join(AV, n)):
                    shutil.copy2(os.path.join(AV, n), os.path.join(AV, n.replace("avaliacao", "avaliacao_motorA_qwen3-8b")))
            rodar("comparar_motores_decs.py", ["--teste", TESTE], amb)
            copiar(ARQS, config.DADOS_LOCAIS, Q8, mover=True)
            log("modelos do 8B em %s" % Q8)
    finally:
        copiar(ARQS, E5, config.DADOS_LOCAIS)                                  # o e5 volta a ser o que esta em uso
        copiar(["avaliacao.md", "avaliacao.json"], os.path.join(AV, "e5-large_atual"), AV)
        log("modelos e5 devolvidos ao lugar (em uso ate a decisao)")
    log("=== FIM ===  resultado: saidas/avaliacao/comparacao_motores_decs.md e revisao_motores_decs_30.xlsx")


if __name__ == "__main__":
    main()
