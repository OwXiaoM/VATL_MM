"""Predeclared validation-only checkpoint criterion for graph reconstruction."""
import math
import hashlib
import json

POLICY={'name':'native_geometry_v1','threshold':.5,'score':'macro_subject_harmonic_mean(pseudo_dice,exact_source_branch_preservation)',
        'tie_break':['lower_macro_cross_source_false_pair_fraction','earlier_step','checkpoint_sha256'],
        'score_round_decimals':12,'selection_subjects':'checkpoint validation people only','evaluation_used':False,
        'scope':'graph-derived pseudo occupancy and source-path geometry; not anatomical radius or image-only prediction'}


def policy_sha256():return hashlib.sha256(json.dumps(POLICY,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def geometric_score(dice,branch):
    if not all(math.isfinite(v) and 0<=v<=1 for v in [dice,branch]):raise ValueError('Finite bounded geometry metrics required')
    return 2*dice*branch/(dice+branch) if dice+branch else 0.


def select_candidate(candidates):
    if not candidates:raise ValueError('No validation candidates')
    for c in candidates:
        if not math.isfinite(c['score']) or not 0<=c['score']<=1:raise ValueError('Invalid candidate score')
        if c['false_pair_fraction'] is not None and not (math.isfinite(c['false_pair_fraction']) and 0<=c['false_pair_fraction']<=1):raise ValueError('Invalid false-pair fraction')
    return sorted(candidates,key=lambda c:(-round(c['score'],12),c['false_pair_fraction'] if c['false_pair_fraction'] is not None else 0.,c['step'],c['checkpoint_sha256']))[0]
