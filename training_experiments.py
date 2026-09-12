"""Experimentos offline de features ou treinamento, sem publicar modelo ativo.

O estudo de features usa apenas development. O estudo de treinamento avalia
o holdout depois de congelar a receita selecionada nas janelas internas.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import config
from feature_engine import FEATURES, EXPERIMENTAL_FEATURES, EXPERIMENTAL_FEATURES_VERSION
from model_contract import compatible, fingerprint, write_json_atomic
from train_model import temporal_partitions, walk_forward_evaluate, choose_candidate


def compare(dataset_path, output_path):
    metadata = json.loads(Path(str(dataset_path)+'.metadata.json').read_text())
    if not compatible(metadata, config.get_active_symbol()) or metadata.get('dataset_sha256') != fingerprint(dataset_path):
        raise ValueError('dataset sem proveniência compatível')
    if metadata.get('experimental_features_version') != EXPERIMENTAL_FEATURES_VERSION:
        raise ValueError('reconstrua o dataset para obter as features experimentais')
    df = pd.read_csv(dataset_path)
    features = FEATURES+EXPERIMENTAL_FEATURES
    if not np.isfinite(df[features].values).all():
        raise ValueError('features experimentais indisponíveis para este timeframe/janela')
    test = temporal_partitions(df)[-1]
    development = df[df.expiry_epoch < int(test.feature_start_epoch.iloc[0])]
    report = {'dataset_sha256': metadata['dataset_sha256'], 'symbol': metadata['symbol'],
              'experiment_version': EXPERIMENTAL_FEATURES_VERSION, 'holdout_evaluated': False,
              'holdout_feature_start': int(test.feature_start_epoch.iloc[0]),
              'development_last_expiry': int(development.expiry_epoch.max()),
              'profiles': {}}
    for name, selected in [('baseline', FEATURES), ('horizon_context', features)]:
        candidates = walk_forward_evaluate(development, selected)
        report['profiles'][name] = {'features': selected, 'candidates': candidates,
                                   'selected': choose_candidate(candidates)['recipe_id']}
    write_json_atomic(output_path, report)
    return report


def compare_training(dataset_path, output_dir, *, search_trials=4, payout_quotes_path=None):
    """Executa receita e baseline em diretório novo, sem acessar o champion."""
    import shutil
    from train_model import train

    # Não sobrescreve relatórios nem usa diretório operacional como destino.
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(dataset_path)
    try:
        # O par CSV/sidecar deve permanecer consistente durante a captura.
        before = fingerprint(source)
        metadata = Path(str(source)+'.metadata.json').read_bytes()
        snapshot = destination / 'dataset.csv'
        shutil.copyfile(source, snapshot)
        if fingerprint(source) != before or fingerprint(snapshot) != before:
            raise ValueError('dataset mudou durante o snapshot')
        if Path(str(source)+'.metadata.json').read_bytes() != metadata:
            raise ValueError('metadados mudaram durante o snapshot')
        Path(str(snapshot)+'.metadata.json').write_bytes(metadata)
        quotes = None
        if payout_quotes_path is not None:
            quotes_hash = fingerprint(payout_quotes_path)
            quotes = destination/'payout_quotes.csv'
            shutil.copyfile(payout_quotes_path, quotes)
            if fingerprint(quotes) != quotes_hash or fingerprint(payout_quotes_path) != quotes_hash:
                raise ValueError('cotações mudaram durante o snapshot')
        # Config global não é alterada: train recebe opção explícita de baseline.
        result = train(snapshot, destination/'model.candidate', search_trials=search_trials,
                       payout_quotes_path=quotes, compare_champion=False)
        result['experiment_only'] = True
        result['promoted'] = False
        write_json_atomic(destination/'comparison.json', result)
        return result
    except Exception:
        # Evidência de falha permanece no diretório; nenhum resultado parcial é publicado.
        (destination/'FAILED').write_text('Experimento incompleto; consulte a exceção da execução.\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, help='arquivo para features; diretório novo para training')
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--study', choices=('features', 'training'), default='features')
    parser.add_argument('--search-trials', type=int, default=4)
    parser.add_argument('--payout-quotes')
    args = parser.parse_args()
    config.set_active_symbol(args.symbol)
    if args.study == 'training':
        compare_training(args.input, args.output, search_trials=args.search_trials, payout_quotes_path=args.payout_quotes)
    else:
        compare(args.input, args.output)
    print(f'Relatório de comparação gravado em {args.output}; nenhum modelo publicado.')
