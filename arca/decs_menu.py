# -*- coding: utf-8 -*-
"""Opcoes D do menu: o fluxo de DECS de ponta a ponta.

Toda opcao que poderia gravar roda em SIMULACAO por padrao e mostra o plano.
Gravar exige confirmacao explicita a cada execucao. Com ARCA_SOMENTE_LEITURA=sim
tudo vira simulacao.
"""
import os
import re
import time
from collections import Counter

from rich import box
from rich.prompt import Confirm, Prompt
from rich.table import Table

from . import config
from . import decs_correcao as C
from . import decs_rastreabilidade as R
from .ui import aviso, console, erro, info, ler_lista_colada, ok, titulo

_CTX = {}


# ---------------------------------------------------------------------------
# Carregamento (uma vez por sessao)
# ---------------------------------------------------------------------------
def contexto(sugerir=True):
    """Vocabulario + acervo + indexador + escolha de galho."""
    if "ctx" not in _CTX:
        from . import decs_acervo, decs_galho, decs_indexador, decs_vocabulario
        t = time.time()
        info("Carregando vocabulario (BIREME + arvore de gravacao: %s)..."
             % ("decs.xml v2.1 com categorias" if config.DECS_ALVO_GRAVACAO == "xml_v2"
                else "copia do servidor"))
        vocab = decs_vocabulario.carregar_vocabulario()
        info("Carregando acervo coletado...")
        acervo = decs_acervo.carregar_acervo()
        info("Carregando indice de documentos e modelo...")
        ix = decs_indexador.montar_indexador(vocab, acervo, com_modelo=True)
        if ix.modelo is None:
            aviso("Modelo ainda nao treinado: rode a opcao D5. Sem ele, o script so "
                  "CORRIGE o que existe e nao sugere descritores novos.")
        galho = decs_galho.EscolhaDeGalho(vocab, acervo, vetores=ix.V, handles=ix.handles)
        if galho.modelo is None:
            aviso("Modelo de galho ainda nao treinado (D5): usando o precedente do acervo.")
        _CTX["ctx"] = C.Contexto(vocab, ix, galho, sugerir=True)
        _CTX["acervo"] = acervo
        ok("Pronto em %.0fs." % (time.time() - t))
    ctx = _CTX["ctx"]
    ctx.sugerir = sugerir and ctx.ix is not None and ctx.ix.modelo is not None
    return ctx, _CTX["acervo"]


def _cliente():
    from . import dspace
    return dspace.conectar(aviso=info)


def _pode_gravar():
    if config.SOMENTE_LEITURA:
        aviso("Modo SOMENTE-LEITURA (ARCA_SOMENTE_LEITURA=sim): o resultado sera uma SIMULACAO.")
        return False
    return Confirm.ask("[bold red]Gravar no ARCA de verdade?[/bold red]", default=False)


# ---------------------------------------------------------------------------
# Exibicao
# ---------------------------------------------------------------------------
_COR = {"MANTER": "dim", "EXPANDIR": "cyan", "ATUALIZAR": "yellow", "DESAMBIGUAR": "yellow",
        "ADICIONAR": "green", "SUGERIR": "blue", "REVISAR": "magenta", "RETIRAR": "red",
        "PARA_PALAVRA_CHAVE": "red", "FORA_DA_ARVORE": "dim"}


def mostrar_plano(handle, linhas):
    tab = Table(title="Plano para %s" % handle, box=box.SIMPLE_HEAVY, show_lines=False)
    tab.add_column("Acao", no_wrap=True)
    tab.add_column("Descritor")
    tab.add_column("Antes")
    tab.add_column("Depois")
    tab.add_column("Como", no_wrap=True)
    for l in linhas:
        cor = _COR.get(l["acao"], "")
        como = l["galho"] or l["metodo"]
        if l["probabilidade"]:
            como += " p=%s" % l["probabilidade"]
        tab.add_row("[%s]%s[/%s]" % (cor, l["acao"], cor), l.get("descritor", ""), l["antes"], l["depois"], como)
    console.print(tab)


# ---------------------------------------------------------------------------
# Fluxos
# ---------------------------------------------------------------------------
def _ler_handles(texto):
    """Separa por linha, tabulacao, virgula, ponto e virgula ou espaco."""
    return [h for h in re.split(r"[\s,;]+", texto.strip()) if h]


def _processar(handles, rotulo, mostrar_cada=False, ao_vivo=False, sugerir=None, gravar=None):
    """Simula (ou grava) o plano de cada item.

    `ao_vivo`: em simulacao, le cada item agora no ARCA (so leitura) em vez de
    usar a copia coletada pelo D4 — para D1/D2, que mexem em poucos itens.
    """
    if sugerir is None:
        sugerir = Confirm.ask("Sugerir/adicionar descritores novos?", default=True)
    ctx, acervo = contexto(sugerir=sugerir)
    if gravar is None:
        gravar = _pode_gravar()
    rel = C.Relatorio(rotulo)
    status = Counter()
    cliente = _cliente() if (gravar or ao_vivo) else None
    t0 = time.time()
    for i, h in enumerate(handles, 1):
        try:
            if gravar:
                st, linhas, final = C.aplicar(cliente, h, ctx)
                status[st] += 1
                rel.escrever(linhas, st == "OK")
            else:
                item = None
                if ao_vivo:
                    obj = cliente.item_por_handle(h)
                    item = C.item_do_dspace(obj) if obj else None
                    h = (obj or {}).get("handle") or h
                else:
                    item = acervo.get(h)
                if item is None:
                    status["NAO_ENCONTRADO" if ao_vivo else "NAO_COLETADO (rode D4)"] += 1
                    continue
                linhas, final, kws = C.planejar(h, item, ctx)
                m = C.mudou(item, final, kws)
                # simulacao fica so no CSV do plano; o SQLite registra o que
                # foi de fato gravado (70 mil simulacoes inchariam o banco)
                rel.escrever(linhas, m)
                status["SIMULADO_COM_MUDANCA" if m else "SIMULADO_SEM_MUDANCA"] += 1
            if mostrar_cada:
                mostrar_plano(h, linhas)
        except Exception as e:  # um item com problema nao para o lote
            status["ERRO"] += 1
            erro("%s: %s" % (h, e))
        if i % 500 == 0:
            info("  %d/%d itens (%.0fs)" % (i, len(handles), time.time() - t0))
    caminho = rel.fechar()
    console.print()
    ok("Plano salvo em %s" % caminho)
    tab = Table(title="Resumo", box=box.SIMPLE)
    tab.add_column("Acao")
    tab.add_column("Valores", justify="right")
    for a, n in rel.contagem.most_common():
        tab.add_row(a, str(n))
    console.print(tab)
    for s, n in status.most_common():
        info("  %s: %d item(ns)" % (s, n))
    est = ctx.galho.estatistica
    if est:
        info("  Galho escolhido por: " + ", ".join("%s %d" % kv for kv in est.most_common()))
    return caminho, rel, status


def _decs_registros(handles):
    """Um so caminho para registros (como as thumbnails): le cada item AGORA no ARCA e faz tudo, com a
    metodologia ja aplicada ao acervo. Pergunta no comeco se grava (e quais sugestoes) e depois trabalha sozinho.

    Item COM DeCS, tres correcoes (ferramentas/): 1) raiz: poe a categoria de topo no galho que veio sem ela
    (corrigir_raiz_decs.py); 2) termo solto: grava o galho completo pela regra hibrida de 2026-10-19
    (galho_termos_soltos.py); 3) tira o que nao tem arvore: descritor de verificacao e qualificador solto
    (retirar_sem_arvore_decs.py). Depois, sugere so o descritor que o modelo recomenda com muita confianca
    (>= limiar automatico, precisao ~90% na avaliacao D5) e que o catalogador nao pos.
    Item SEM DeCS: sugere descritores (limiar de revisao, precisao ~70%; os de alta confianca vem marcados).
    """
    from . import preparo
    if preparo.garantir(aviso=info):
        erro("Ha dados locais que o programa nao conseguiu refazer sozinho (veja acima). Nada foi feito.")
        return
    cli = _cliente()
    com, sem = [], []
    for ident in handles:
        obj = cli.item_por_handle(ident)
        if not obj:
            aviso("%s: item nao encontrado no ARCA" % ident)
            continue
        h = obj.get("handle") or ident
        (com if (obj.get("metadata") or {}).get(config.CAMPO_DECS) else sem).append(h)
    info("%d registro(s) com DeCS, %d sem DeCS." % (len(com), len(sem)))
    # as perguntas vem ANTES do processamento (pedido do responsavel, 2026-10-20: processar e demorado; responder
    # no comeco e deixar o programa trabalhar sozinho ate o fim)
    # D1/D2 sempre gravam as correcoes: escolher a opção e a confirmação (decisao do responsavel, 2026-10-20).
    # Para testar sem gravar: ARCA_SOMENTE_LEITURA=sim.
    gravar = not config.SOMENTE_LEITURA
    if not gravar:
        aviso("Modo SOMENTE-LEITURA (ARCA_SOMENTE_LEITURA=sim): so o plano, nada sera gravado.")
    esc = "nenhuma"
    if gravar:
        info("Sugestoes de descritores novos: adicionar = so as muito recomendadas (p >= limiar automatico, precisao "
             "~90%%); todas = inclui as possiveis (~70%%); escolher = pergunta uma a uma no fim; nenhuma.")
        esc = Prompt.ask("Gravar quais sugestoes?", choices=["adicionar", "todas", "escolher", "nenhuma"], default="nenhuma")

    # ---- 1. correcoes nos registros com DeCS ---------------------------------------------
    etapas = [("0/3 Termos que nao sao DeCS: vao para palavra-chave (ou convertem, se for o mesmo conceito)",
               "corrigir_termos_fora_do_decs.py", []),
              ("1/3 Raiz: categoria de topo e nos renomeados no DeCS 2026", "corrigir_raiz_decs.py", []),
              ("2/3 Termos soltos: galho completo (regra hibrida aplicada ao acervo)", "galho_termos_soltos.py",
               ["--completar"]),
              ("3/3 Termos sem arvore: descritor de verificacao e qualificador solto", "retirar_sem_arvore_decs.py", [])]
    if com:
        titulo("%s (%d registro(s) com DeCS)" % ("Gravando as correcoes" if gravar else "Plano das correcoes (nada gravado)",
                                               len(com)))
        for nome, script, extra in etapas:
            info(nome)
            _rodar_ferramenta(script, ["--item"] + com + extra + (["--gravar"] if gravar else []))

    # ---- 2. sugestoes (lidas de novo: depois das correcoes) ------------------------------
    ctx, _ = contexto(sugerir=True)
    if not ctx.sugerir:
        aviso("Modelo de sugestao nao carregado (D5): sem sugestoes de descritores.")
        return
    planos = []
    for h in com + sem:
        obj = cli.item_por_handle(h)
        tem = bool((obj.get("metadata") or {}).get(config.CAMPO_DECS))
        linhas, _final, _kws = C.planejar(h, C.item_do_dspace(obj), ctx)
        novas = [l for l in linhas if l["acao"] == "ADICIONAR" or (not tem and l["acao"] == "SUGERIR")]
        if novas:
            titulo("Sugestoes para %s (%s)" % (h, "tem DeCS: so os muito recomendados que faltam" if tem
                                              else "sem DeCS"))
            mostrar_plano(h, novas)
            planos.append((h, tem, novas))
    if not planos:
        info("Nenhuma sugestao de descritor novo.")
        return
    if not gravar:
        info("Sugestoes nao gravadas.")
        return
    for h, tem, novas in planos:
        # palavra-chave do autor que ja e DeCS entra sempre (regra do responsavel, 2026-10-20; DECISOES §18);
        # a escolha do inicio vale so para as sugestoes do modelo
        kw = [l for l in novas if l["metodo"] == "palavra_chave_autor"]
        novas = [l for l in novas if l["metodo"] != "palavra_chave_autor"]
        if esc == "nenhuma":
            sel = []
        elif esc == "adicionar":
            sel = [l for l in novas if l["acao"] == "ADICIONAR"]
        elif esc == "todas":
            sel = novas
        else:
            sel = [l for l in novas if Confirm.ask("%s: gravar %s (p=%s)?" % (h, l["descritor"], l["probabilidade"]),
                                                   default=False)]
        if kw + sel:
            info("%s: %s" % (h, C.gravar_novos(cli, h, kw + sel, ctx, acrescentar=tem)))


def d1_um_item():
    titulo("D1 - DeCS de um registro (corrige o que tem; sugere se nao tem)")
    _decs_registros([Prompt.ask("Handle, UUID ou URL do item").strip()])


def d2_lista():
    titulo("D2 - DeCS de uma lista de registros (corrige o que tem; sugere se nao tem)")
    colado = ler_lista_colada("Cole os handles, UUIDs ou URLs (um por linha, ou separados por "
                              "tabulacao, virgula, ponto e virgula ou espaco) OU o caminho de um "
                              ".txt. Linha vazia termina.")
    caminho = " ".join(colado).strip('"') if len(colado) <= 3 else ""
    if caminho and os.path.isfile(caminho):
        with open(caminho, encoding="utf-8") as f:
            handles = _ler_handles(f.read())
    else:
        handles = colado
    info("%d item(ns)." % len(handles))
    _decs_registros(handles)


def d4_coletar():
    titulo("D4 - Coletar o acervo (somente leitura)")
    from . import decs_coleta
    r = decs_coleta.coletar(_cliente(), progresso=info)
    ok("Acervo: %(depois)d itens (%(novos)d novos, %(atualizados)d atualizados, "
       "%(com_decs)d com DECS)." % r)
    _CTX.clear()


def d5_treinar():
    titulo("D5 - Treinar e avaliar o modelo de DECS")
    import runpy
    runpy.run_path(os.path.join(config.RAIZ, "ferramentas", "decs", "treinar_avaliar.py"), run_name="__main__")
    _CTX.clear()
    ok("Relatorio: %s" % os.path.join(config.SAIDAS_AVALIACAO, "avaliacao.md"))


def d6_baixar_vocabulario():
    titulo("D6 - Baixar o vocabulario decs publicado no servidor (somente leitura)")
    import runpy
    runpy.run_path(os.path.join(config.RAIZ, "ferramentas", "decs", "baixar_vocabulario_servidor.py"),
                   run_name="__main__")
    _CTX.clear()


def d7_gerar_xml():
    titulo("D7 - Gerar decs.xml (DeCS BIREME) para publicar no DSpace")
    import runpy
    runpy.run_path(os.path.join(config.RAIZ, "ferramentas", "decs", "gerar_vocabulario_dspace.py"),
                   run_name="__main__")


def d0_restaurar():
    titulo("D0 - Restaurar o DECS original do catalogador")
    h = Prompt.ask("Handle, UUID ou URL do item").strip()
    if config.SOMENTE_LEITURA:
        aviso("Modo SOMENTE-LEITURA: vou so MOSTRAR o que seria restaurado.")

    def confirmar(atuais, original, fonte):
        console.print("[bold]Origem do backup:[/bold] %s" % fonte)
        console.print("[red]Agora:[/red] %s" % (atuais or "(vazio)"))
        console.print("[green]Original:[/green] %s" % (original or "(vazio)"))
        return (not config.SOMENTE_LEITURA) and Confirm.ask("Restaurar?", default=False)

    st, _a, _o = C.restaurar(_cliente(), h, confirmar)
    info("Resultado: %s" % st)


def r1_rastreabilidade():
    titulo("R1 - Rastreabilidade")
    tab = Table(box=box.SIMPLE)
    for c in ("modelo", "aplicado", "registros", "primeiro", "ultimo"):
        tab.add_column(c)
    rotulo = {1: "sim", 0: "simulacao", -1: "antigo (?)"}
    for m, a, n, p, u in R.estatisticas():
        tab.add_row(m, rotulo.get(a, str(a)), str(n), str(p)[:16], str(u)[:16])
    console.print(tab)
    rep = R.reprocessaveis()
    caminho = os.path.join(config.SAIDAS_DECS, "reprocessar.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    info("%d item(ns) gravados com versao anterior do metodo -> %s (use na D2)"
         % (len(rep), caminho))


# -- galhos e termos soltos: gravam no ARCA com confirmacao (mesmo fluxo da gravacao do acervo) ------------
_RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rodar_ferramenta(script, args):
    """Roda uma ferramenta do projeto e mostra a saida no menu."""
    import subprocess
    import sys
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    subprocess.run([sys.executable, os.path.join(_RAIZ_PROJETO, "ferramentas", "decs", script)] + args,
                   cwd=_RAIZ_PROJETO, env=env)


OPCOES = [
    ("D1", "DeCS - Um registro", "Corrige e completa o DeCS de um item (raiz, galho, termos sem arvore) e sugere o que falta. Mostra tudo antes; so grava se voce confirmar", d1_um_item),
    ("D2", "DeCS - Lista de registros", "O mesmo do D1 para varios itens (cole handles/links ou o caminho de um .txt)", d2_lista),
    ("D0", "DeCS - Desfazer", "Volta o DeCS de um item ao que o catalogador tinha ([DECS-ORIGINAL])", d0_restaurar),
    ("R1", "DeCS - Historico", "O que o programa gravou em cada item", r1_rastreabilidade),
    ("D4", "Manut. - Coletar acervo", "Atualiza a copia local usada so para treinar o modelo (o D1/D2 nao precisa)", d4_coletar),
    ("D5", "Manut. - Treinar modelo", "Retreina e mede o modelo de sugestao (raro; leva horas)", d5_treinar),
    ("D6", "Manut. - Vocabulario", "Baixa a arvore decs publicada no DSpace (quando o vocabulario mudar)", d6_baixar_vocabulario),
    ("D7", "Manut. - Gerar decs.xml", "XML do DeCS 2026 para o administrador publicar", d7_gerar_xml),
]


