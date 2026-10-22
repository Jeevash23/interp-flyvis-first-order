"""Independent, read-only numerical audit of the post-run reporting supplement.

No model is imported or simulated. The frozen analysis and its outputs remain
unchanged; this script writes only its own verification record.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from scipy.stats import spearmanr

P = Path(__file__).resolve().parents[2]
R = P / '04-Results/stage1'
D = P / '03-Data/stage1'
hashes = {}
cache = {}


def sha(path):
    path = Path(path)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    hashes[str(path.relative_to(P))] = digest
    return digest


def read_json(path):
    sha(path)
    return json.loads(path.read_text())


def bank(path):
    """Hash each source and independently check its saved checksum once."""
    path = Path(path)
    if path not in cache:
        meta = read_json(path.with_suffix('.json'))
        assert sha(path) == meta['sha256'], f'Checksum mismatch: {path}'
        with np.load(path) as z:
            cache[path] = {k: z[k] for k in ['dsi', 'denominator',
                            'synapse_count', 'current_score', 'gradient']
                           if k in z.files}
    return cache[path]


def scalar_close(actual, expected, tolerance=2e-6):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return 0.0
    difference = abs(float(actual) - float(expected))
    assert difference <= tolerance, (actual, expected, difference)
    return difference


def finite_mean(values):
    return float(np.mean(values)) if values else None


sup = read_json(R / 'analysis/reporting-supplement.json')
names = read_json(D / 'network-structure.json')['model_types']
sha(Path(__file__).resolve())
permutation_rows = []
permutation_max_error = 0.0
methods = ['first_order', 'synapse_count', 'native_current']
for split, models, keys in [
    ('development', ['000', '004'], ['dev_slow', 'dev_fast']),
    ('final', ['001', '005'], ['final_slow', 'final_fast']),
]:
    for model in models:
        for strength in [0.5, 1.0]:
            observed = {method: [] for method in methods}
            complete = 0
            for key in keys:
                folder = R / split / model / key
                base = bank(folder / 'baseline.npz')
                gradient = bank(folder / 'gradient.npz')['gradient']
                singles = [bank(folder / f's{int(strength*100):03d}-cell{i:02d}.npz')
                           for i in range(len(names))]
                dsi = np.stack([row['dsi'] for row in singles], axis=1)
                den = np.stack([row['denominator'] for row in singles], axis=1)
                effect = dsi - base['dsi'][:, None]
                prediction = strength * gradient
                valid = ((den > 1e-8) & (base['denominator'][:, None] > 1e-8)
                         & np.isfinite(effect) & np.isfinite(prediction))
                for readout in range(8):
                    if not valid[readout].all():
                        continue
                    complete += 1
                    scores = {
                        'first_order': abs(prediction[readout]),
                        'synapse_count': base['synapse_count'][readout],
                        'native_current': base['current_score'][readout],
                    }
                    truth = abs(effect[readout])
                    for method, score in scores.items():
                        if np.ptp(score) > 0 and np.ptp(truth) > 0:
                            value = float(spearmanr(score, truth).statistic)
                            if np.isfinite(value):
                                observed[method].append(value)
            for method in methods:
                saved = next(row for row in sup['permutation_comparisons']
                             if row['model'] == model and row['strength'] == strength
                             and row['method'] == method)
                assert saved['complete_units'] == complete
                assert saved['estimable_complete_units'] == len(observed[method])
                value = finite_mean(observed[method])
                error = scalar_close(value, saved['observed_mean_spearman_same_units'], 1e-12)
                permutation_max_error = max(permutation_max_error, error)
                permutation_rows.append({
                    'model': model, 'strength': strength, 'method': method,
                    'complete_units': complete,
                    'estimable_complete_units': len(observed[method]),
                    'independent_observed_mean_spearman': value,
                    'absolute_error': error,
                })

biology_rows = []
biology_max_error = 0.0
for row in sup['biology']:
    folder = R / row['split'] / row['model'] / row['stimulus']
    base = bank(folder / 'baseline.npz')
    for label, code in [('Mi1', f'cell{names.index("Mi1"):02d}'),
                        ('Tm3', f'cell{names.index("Tm3"):02d}'),
                        ('Mi1+Tm3', 'pair0')]:
        intervention = bank(folder / f's{int(row["strength"]*100):03d}-{code}.npz')
        valid = ((base['denominator'] > 1e-8)
                 & (intervention['denominator'] > 1e-8)
                 & np.isfinite(base['dsi']) & np.isfinite(intervention['dsi']))
        indices = np.flatnonzero(valid[:4])
        effect = intervention['dsi'].astype(np.float64) - base['dsi'].astype(np.float64)
        value = float(effect[indices].mean()) if len(indices) else None
        saved = row['effects'][label]
        assert saved['dsi_valid_mask'] == valid.tolist()
        assert saved['valid_T4_count'] == len(indices)
        error = scalar_close(value, saved['mean_valid_T4_delta'], 1e-7)
        biology_max_error = max(biology_max_error, error)
        biology_rows.append({
            'model': row['model'], 'strength': row['strength'],
            'stimulus': row['stimulus'], 'intervention': label,
            'valid_mask': valid.tolist(), 'valid_T4_count': len(indices),
            'independent_mean_valid_T4_delta_float64': value,
            'absolute_error': error,
        })

pair_names = [('Mi1', 'Tm3'), ('Mi1', 'Mi4'), ('Tm3', 'Mi4'), ('Mi4', 'Mi9'),
              ('Tm1', 'Tm2'), ('Tm1', 'Tm9'), ('Tm2', 'Tm4'), ('Tm4', 'Tm9')]
pair_rows = []
pair_max_error = 0.0
eligible_total = 0
hit_total = 0
qualifying_pairs = set()
for row in sup['pair_interactions']:
    folder = R / row['split'] / row['model'] / row['stimulus']
    prefix = f's{int(row["strength"]*100):03d}'
    first, second = row['pair']
    sources = [folder / 'baseline.npz',
               folder / f'{prefix}-cell{names.index(first):02d}.npz',
               folder / f'{prefix}-cell{names.index(second):02d}.npz',
               folder / f'{prefix}-pair{pair_names.index((first, second))}.npz']
    base, a, b, joint = [bank(path) for path in sources]
    valid = np.logical_and.reduce([
        (source['denominator'] > 1e-8) & np.isfinite(source['dsi'])
        for source in [base, a, b, joint]])
    contrast = (joint['dsi'].astype(np.float64) - a['dsi'].astype(np.float64)
                - b['dsi'].astype(np.float64) + base['dsi'].astype(np.float64))
    maximum = float(abs(contrast[valid]).max()) if valid.any() else None
    hits = int((abs(contrast[valid]) >= 0.01).sum())
    assert row['dsi_eligible_mask'] == valid.tolist()
    assert row['dsi_eligible_count'] == int(valid.sum())
    assert row['dsi_at_least_001_count'] == hits
    errors = [scalar_close(maximum, row['max_abs_dsi'])]
    for value, is_valid, saved in zip(contrast, valid, row['dsi_interaction_valid']):
        errors.append(scalar_close(float(value) if is_valid else None, saved))
    pair_max_error = max(pair_max_error, *errors)
    eligible_total += int(valid.sum())
    hit_total += hits
    if hits:
        qualifying_pairs.add((first, second))
    pair_rows.append({
        'model': row['model'], 'strength': row['strength'],
        'stimulus': row['stimulus'], 'pair': [first, second],
        'eligible_count': int(valid.sum()), 'threshold_count': hits,
        'independent_max_abs_dsi_float64': maximum,
        'maximum_absolute_error': max(errors),
    })

# Bounded diagnostic of the reported largest raw-mean contrast, using saved traces.
folder = R / 'development/004/dev_slow'
paths = [folder / 'baseline.npz',
         folder / f's100-cell{names.index("Mi4"):02d}.npz',
         folder / f's100-cell{names.index("Mi9"):02d}.npz',
         folder / 's100-pair3.npz']
raw = []
for path in paths:
    bank(path)
    with np.load(path) as z:
        raw.append({'trace': z['trace'], 'raw_mean': z['raw_mean']})
speed = 9.5
endpoint = ((10*2.25/5.8 + 2.25/5.8) - (-10*2.25/5.8)) / speed
times = (np.arange(raw[0]['trace'].shape[1], dtype=np.float64) + 200) * .005 - 1
window = (times >= 0) & (times <= endpoint)
contrast = (raw[3]['raw_mean'].astype(float) - raw[1]['raw_mean'].astype(float)
            - raw[2]['raw_mean'].astype(float) + raw[0]['raw_mean'].astype(float))
index = np.unravel_index(abs(contrast).argmax(), contrast.shape)
raw_rows = []
for label, source in zip(['baseline', 'Mi4', 'Mi9', 'Mi4+Mi9'], raw):
    trace = source['trace']
    assert np.isfinite(trace).all()
    raw_rows.append({
        'condition': label, 'all_saved_trace_values_finite': True,
        'scoring_window_max_abs_voltage': float(abs(trace[:, window]).max()),
        'full_saved_trial_max_abs_voltage': float(abs(trace).max()),
        'last_saved_frame_max_abs_voltage': float(abs(trace[:, -1]).max()),
        'raw_mean_at_maximum_contrast_index': float(source['raw_mean'][index]),
        'trace_recomputed_float64_raw_mean_at_index':
            float(trace[index[0], window, index[1]].astype(float).mean()),
    })
stimulus_path = D / 'stimulus-dev_slow.npz'
sha(stimulus_path)
with np.load(stimulus_path) as stimulus:
    angle = int(stimulus['angle'][index[0]])
    polarity = int(stimulus['intensity'][index[0]])
amplification = {
    'model': '004', 'stimulus': 'dev_slow', 'strength': 1.0,
    'pair': ['Mi4', 'Mi9'], 'readout': 'T5c',
    'trajectory_index': int(index[0]), 'readout_index': int(index[1]),
    'angle_degrees': angle, 'intensity': polarity,
    'scoring_window_frames': int(window.sum()),
    'last_scoring_frame_time_seconds': float(times[window][-1]),
    'saved_post_onset_frames': len(times),
    'independent_max_abs_raw_mean_contrast_float64': float(abs(contrast[index])),
    'conditions': raw_rows,
    'interpretation': 'Severe intervention-driven amplification is visible within '
                      'the scoring window and grows through the saved postperiod. '
                      'Finite arrays do not demonstrate stability or physiological '
                      'plausibility. No new simulation or stability experiment was run.',
}
assert len(permutation_rows) == 24
assert len(biology_rows) == 48
assert len(pair_rows) == 128
assert eligible_total == 900 and hit_total == 511 and len(qualifying_pairs) == 8
record = {
    'status': 'pass', 'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Local AI-assisted verification, not external scientific review. '
             'Read-only recomputation from saved NPZ arrays; no simulations, '
             'GPU allocations, new threshold selection or frozen-code changes. '
             'Permutation coverage and observed means were independently checked; '
             'the 1000 permutation distributions were not regenerated by this audit.',
    'source_sha256': hashes,
    'counts': {'permutation_comparisons': len(permutation_rows),
               'biology_interventions': len(biology_rows),
               'pair_panel_rows': len(pair_rows),
               'nominal_pair_dsi': len(pair_rows)*8,
               'eligible_pair_dsi': eligible_total,
               'pair_dsi_at_least_001': hit_total,
               'qualifying_pairs': len(qualifying_pairs)},
    'maximum_absolute_discrepancies': {
        'permutation_observed_mean_spearman': permutation_max_error,
        'biology_valid_T4_mean_float64_vs_saved': biology_max_error,
        'pair_dsi_float64_vs_saved': pair_max_error},
    'permutation_comparisons': permutation_rows,
    'biology_interventions': biology_rows,
    'pair_panel_checks': pair_rows,
    'qualifying_pairs': [list(pair) for pair in sorted(qualifying_pairs)],
    'raw_amplification': amplification,
}
output = R / 'verification/reporting-supplement-audit.json'
output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: record[key] for key in ['status', 'counts',
                  'maximum_absolute_discrepancies']}, indent=2))
