"""Selection score of the final model on the German grid (Section 5.1: 0.137 against 0.186).

Applies the score of the forward selection (Equation 24; eight selection statistics, penalty
0.005 per added term) to the samples of the final model. On the 1,100 samples of the final model the
penalized score is 0.1367. Note that the score 0.186 of the selected model (sigma^2 + t1) was
computed on 50 samples during the selection.

Input: output/samples/final/samples_DE.pkl
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.grid import build_country_graph
from common.paths import OUTPUT_DIR
from forward_selection import LAMBDA_PEN, score, selection_statistics

if __name__ == '__main__':
    target = selection_statistics(build_country_graph('DE'))
    with open(OUTPUT_DIR / 'samples' / 'final' / 'samples_DE.pkl', 'rb') as f:
        samples = pickle.load(f)
    raw = score(samples, target)
    print(f'N = {len(samples)}: raw score = {raw:.4f}, penalized (3 added terms) = {raw + 3 * LAMBDA_PEN:.4f}')
