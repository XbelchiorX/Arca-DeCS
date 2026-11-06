# -*- coding: utf-8 -*-
"""Saida no console (rich), compartilhada por todos os modulos."""
import os
import sys

# O console do Windows abre em cp1252 e nao sabe escrever os simbolos do rich
# (UnicodeEncodeError: 'charmap'); pior, o erro acontecia dentro de erro() e o
# proprio aviso de falha sumia. Reconfigurar aqui cobre o projeto inteiro.
for _fluxo in ("stdout", "stderr"):
    try:
        getattr(sys, _fluxo).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console  # noqa: E402

console = Console()


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def titulo(texto):
    console.print()
    console.rule("[bold cyan]%s[/bold cyan]" % texto)


def ok(texto):
    console.print("[bold green]✅ %s[/bold green]" % texto)


def aviso(texto):
    console.print("[bold yellow]⚠️  %s[/bold yellow]" % texto)


def erro(texto):
    console.print("[bold red]❌ %s[/bold red]" % texto)


def info(texto):
    console.print("[cyan]%s[/cyan]" % texto)



def ler_lista_colada(mensagem="Cole os identificadores (um por linha, ou separados por "
                              "tabulacao, virgula, ponto e virgula ou espaco). Linha vazia termina."):
    """Le uma lista colada no terminal, com quantas linhas vierem.

    Le linha a linha ate uma linha vazia: colar uma coluna do Excel (tabulada)
    ou uma lista de URLs funciona. Antes a leitura pegava so a 1a linha, e as
    demais "vazavam" para a pergunta seguinte.
    """
    import re
    console.print("[cyan]%s[/cyan]" % mensagem)
    linhas = []
    while True:
        try:
            linha = input("> ")
        except EOFError:
            break
        if not linha.strip():
            break
        linhas.append(linha)
    return [x for x in re.split(r"[\s,;]+", "\n".join(linhas)) if x]
