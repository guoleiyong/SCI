# -*- coding: utf-8 -*-
"""只跑 E12（弃权判据验证）+ 真实数据 MRI/voice 转移，不重复 E0--E11。
用法（在项目根目录 M:\0718 下）：
    python P32/P32-01/run_e12_only.py --out P32/P32-01/results_real
数据路径与 SCI_v4.py 相同（默认 M:\MRI\...、M:\Audio 等），
需要覆盖时加 --mri_path / --voice_path（写法与 SCI_v4.py 一致）。
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import SCI_v4 as s

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results_real")
ap.add_argument("--mri_path", default=None)
ap.add_argument("--voice_path", default=None)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

print("==== E12 only: abstention predictability (real) ====")
s.exp_abstention_sim(a.out, reps=4)          # 模拟参照网格
try:
    s.exp_mri(a.out, mode="real", path=a.mri_path, reps=5 if a.quick else 10)
except FileNotFoundError as e:
    print(f"[skip] E2 MRI: {e}")
try:
    s.exp_voice(a.out, mode="real", path=a.voice_path)
except FileNotFoundError as e:
    print(f"[skip] E3 voice: {e}")
s.exp_abstention_predictability(a.out)       # 汇总 + fig8
print("done -> E12_predictability_real.csv, figures/fig8_abstention_prediction.png")
