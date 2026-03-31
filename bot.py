#!/home/stanis/projetos/Binarias/Deriv/.venv/bin/python3
"""
bot.py — orquestrador principal (entry point).

Uso:
    python3 bot.py                   # modo definido em config.py (DEMO_MODE)
    python3 bot.py --real            # força modo real (pede confirmação)
    python3 bot.py --demo            # força modo demo (ignora config.DEMO_MODE)
    python3 bot.py --balance 500     # saldo inicial para controle de risco

    Coletor (processo separado):
    python3 collector.py             # salva ticks em ticks.csv
"""

import argparse
import signal
import sys

from config import DEMO_MODE
from executor import DerivBot
from risk_manager import RiskManager


def _handle_interrupt(sig, frame) -> None:
    print("\n\n[BOT] Encerrado pelo usuário. Até logo.")
    sys.exit(0)


def _confirm_real_mode() -> bool:
    """Exige confirmação explícita antes de operar com dinheiro real."""
    print("\n" + "!" * 55)
    print("  ATENÇÃO: Modo REAL ativado.")
    print("  Este bot usará DINHEIRO REAL na sua conta Deriv.")
    print("  Certifique-se de que o TOKEN em config.py é de")
    print("  conta REAL e tem permissões de 'trade'.")
    print("!" * 55)
    answer = input("\nDigite 'sim' para confirmar a operação real: ").strip().lower()
    return answer == "sim"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deriv Trading Bot — estratégia multi-indicador profissional.\n"
            "Por padrão opera em modo demo (não requer confirmação)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--real",
        action="store_true",
        help="Força modo real (dinheiro real). Requer confirmação.",
    )
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Força modo demo (ignora config.DEMO_MODE).",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        metavar="USD",
        help=(
            "Saldo inicial para o RiskManager calcular stakes e limites diários. "
            "Use o saldo real da sua conta para cálculos corretos. "
            "(padrão: 1000.0)"
        ),
    )
    args = parser.parse_args()

    # Determina modo: flag CLI tem prioridade sobre config.DEMO_MODE
    if args.real:
        is_demo = False
    elif args.demo:
        is_demo = True
    else:
        is_demo = DEMO_MODE  # lê de config.py

    if not is_demo:
        if not _confirm_real_mode():
            print("Operação cancelada.")
            sys.exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)

    risk_manager = RiskManager(initial_balance=args.balance)
    bot = DerivBot(risk_manager=risk_manager, demo=is_demo)
    bot.run()


if __name__ == "__main__":
    main()
