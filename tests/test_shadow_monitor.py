from shadow_monitor import ShadowMonitor


def monitor(path, model_id='model-a'):
    return ShadowMonitor(
        path,
        symbol='R_100',
        model_id=model_id,
        timeframe_sec=900,
        duration_sec=900,
    )


def test_shadow_requires_30_results_and_55_percent_accuracy(tmp_path):
    tracker = monitor(tmp_path / 'shadow.json')
    start = 1_800_000_000

    for index in range(29):
        entry = start + index * 900
        direction = 1 if index < 17 else 0
        assert tracker.observe_prediction(
            entry_epoch=entry,
            entry_price=100,
            direction=direction,
            confidence=.8,
            net_payout=.70,
        ) == 'pending'
        result = tracker.settle_tick(
            epoch=entry + 900,
            price=101 if index < 17 else 101,
        )
        assert result != 'approved'

    entry = start + 29 * 900
    assert tracker.observe_prediction(
        entry_epoch=entry,
        entry_price=100,
        direction=0,
        confidence=.8,
        net_payout=.70,
    ) == 'pending'
    assert tracker.settle_tick(epoch=entry + 900, price=101) == 'approved'
    assert tracker.result_count == 30
    assert tracker.accuracy == 17 / 30


def test_shadow_rejects_low_payout_and_does_not_count_gaps_or_ties(tmp_path):
    tracker = monitor(tmp_path / 'shadow.json')
    entry = 1_800_000_000
    common = dict(entry_price=100, direction=1, confidence=.8)

    assert tracker.observe_prediction(entry_epoch=entry, net_payout=.69, **common) == 'payout_below_floor'
    assert tracker.observe_prediction(entry_epoch=entry, net_payout=.70, **common) == 'pending'
    assert tracker.settle_tick(epoch=entry + 900 + 11, price=101) == 'expired_with_gap'
    assert tracker.result_count == 0

    assert tracker.observe_prediction(entry_epoch=entry + 1800, net_payout=.70, **common) == 'pending'
    assert tracker.settle_tick(epoch=entry + 2700, price=100) == 'tie'
    assert tracker.result_count == 0


def test_shadow_state_persists_and_new_model_starts_fresh(tmp_path):
    path = tmp_path / 'shadow.json'
    first = monitor(path)
    entry = 1_800_000_000
    assert first.observe_prediction(
        entry_epoch=entry,
        entry_price=100,
        direction=1,
        confidence=.8,
        net_payout=.75,
    ) == 'pending'
    assert first.settle_tick(epoch=entry + 900, price=101) == 'settled'

    resumed = monitor(path)
    assert resumed.result_count == 1
    changed = monitor(path, model_id='model-b')
    assert changed.result_count == 0
    assert changed.state['archives'][0]['model_id'] == 'model-a'


def test_shadow_rejects_bundle_after_30_results_below_accuracy_gate(tmp_path):
    tracker = monitor(tmp_path / 'shadow.json')
    start = 1_800_000_000
    for index in range(30):
        entry = start + index * 900
        assert tracker.observe_prediction(
            entry_epoch=entry,
            entry_price=100,
            direction=1,
            confidence=.8,
            net_payout=.70,
        ) == 'pending'
        result = tracker.settle_tick(epoch=entry + 900, price=99)

    assert result == 'rejected'
    assert tracker.rejected
    assert tracker.state['rejected_at_epoch'] == entry + 900
    assert tracker.observe_prediction(
        entry_epoch=entry + 1800,
        entry_price=100,
        direction=1,
        confidence=.8,
        net_payout=.70,
    ) == 'already_pending_or_approved'