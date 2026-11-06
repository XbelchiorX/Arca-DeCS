# -*- coding: utf-8 -*-
"""Menu do Arca DeCS. As opcoes que gravam no repositorio pedem confirmacao."""
from rich import box
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from . import config, decs_menu as M
from .ui import aviso, clear, console, erro, info, ok


def _tabela():
    t = Table(box=box.ROUNDED, header_style="bold white", border_style="#3d3d5c", padding=(0, 1))
    t.add_column("#", justify="center", style="bold cyan", width=3)
    t.add_column("Opcao", style="bold", width=28)
    t.add_column("O que faz", style="#cccccc")
    t.add_section()
    t.add_row("", "[bold #b98dff]Arca DeCS[/bold #b98dff]", "[#a0a0a0]Corrige o DeCS do catalogador e atribui o que falta, na arvore do DSpace[/#a0a0a0]")
    for cod, nome, desc, _f in M.OPCOES:
        t.add_row(cod, nome, desc)
    t.add_section()
    t.add_row("T", "Testar login", "Abre/valida a sessao no repositorio")
    t.add_row("S", "Sair", "")
    return t


def menu():
    funcoes = {c: f for c, _n, _d, f in M.OPCOES}
    escolhas = list(funcoes) + ["T", "S"]
    while True:
        clear()
        modo = ("[bold yellow]SOMENTE-LEITURA (nada e gravado)[/bold yellow]"
                if config.SOMENTE_LEITURA else "[bold red]ESCRITA LIBERADA[/bold red]")
        console.print(Panel.fit("[bold white]  Arca DeCS  [/bold white]\n%s\n[#808080]Servidor: %s[/#808080]"
                                % (modo, config.BASE_URL), border_style="#4a9eff", padding=(1, 2)))
        console.print(_tabela())
        op = Prompt.ask("Escolha", choices=escolhas + [e.lower() for e in escolhas], default="S").upper()
        try:
            if op == "S":
                return
            if op == "T":
                from . import dspace
                ok("Login OK em %s" % dspace.conectar(aviso=info).base)
            else:
                funcoes[op]()
        except KeyboardInterrupt:
            aviso("Interrompido.")
        except Exception as e:
            erro("%s: %s" % (type(e).__name__, e))
        Prompt.ask("\n[dim]Enter para voltar ao menu[/dim]", default="")
