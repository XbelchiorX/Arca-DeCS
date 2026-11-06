# -*- coding: utf-8 -*-
"""Monitor de progresso para tarefas longas (pedido do responsavel, 2026-10-25: "eu gosto de acompanhar os dados
sendo processados e o tempo restante"). So le: conta as linhas de um arquivo que a tarefa vai escrevendo (backup,
CSV, log) e mostra barra, ritmo, tempo restante e hora prevista de termino. Fechar o monitor nao para a tarefa.

Uso: python ferramentas/monitor_progresso.py <arquivo> <total> ["titulo"] [--cabecalho N]
  ex.: python ferramentas/monitor_progresso.py saidas/palavras_chave/backup_forma_e_caixa_20261025_1128.csv 43182
       "Palavras-chave: gravacao no Arca" --cabecalho 1
"""
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table


def contar(arq):
    try:
        with open(arq, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    arq, total = args[0], int(args[1])
    titulo = args[2] if len(args) > 2 else os.path.basename(arq)
    cab = int(sys.argv[sys.argv.index("--cabecalho") + 1]) if "--cabecalho" in sys.argv else 0
    hist = deque(maxlen=60)                       # (tempo, feitos) dos ultimos ~5 min, para o ritmo
    console = Console()
    with Live(console=console, refresh_per_second=2) as live:
        while True:
            feitos = max(0, contar(arq) - cab)
            agora = time.time()
            hist.append((agora, feitos))
            t0, f0 = hist[0]
            ritmo = (feitos - f0) / (agora - t0) * 60 if agora > t0 and feitos > f0 else 0.0
            falta = max(0, total - feitos)
            eta = timedelta(minutes=falta / ritmo) if ritmo > 0 else None
            tab = Table.grid(padding=(0, 2))
            tab.add_row(ProgressBar(total=total, completed=min(feitos, total), width=60),
                        "[bold]%.1f%%[/bold]" % (100 * feitos / max(1, total)))
            info = Table.grid(padding=(0, 3))
            info.add_row("Feitos: [bold]%d[/bold] de %d" % (feitos, total), "Faltam: [bold]%d[/bold]" % falta)
            info.add_row("Ritmo: [bold]%.0f[/bold] por minuto" % ritmo,
                         "Tempo restante: [bold]%s[/bold]" % (str(eta).split(".")[0] if eta else "calculando..."))
            info.add_row("Agora: %s" % datetime.now().strftime("%H:%M:%S"),
                         "Termina por volta de: [bold green]%s[/bold green]"
                         % ((datetime.now() + eta).strftime("%H:%M") if eta else "..."))
            grade = Table.grid()
            grade.add_row(tab)
            grade.add_row("")
            grade.add_row(info)
            if feitos >= total:
                grade.add_row("\n[bold green]Concluido![/bold green]")
            live.update(Panel(grade, title="[bold]%s[/bold]" % titulo, border_style="#22966e", padding=(1, 2)))
            if feitos >= total:
                break
            time.sleep(5)
    input("\nEnter para fechar o monitor")


if __name__ == "__main__":
    main()
