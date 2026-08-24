"""
予想と結果の照合

predictions_*.csv(生成時に保存した全馬分)と確定成績を突き合わせ、
絞り込み度合いを変えながら成績を出す。

買い目は上位2%で固定して運用するが、全馬分を保存しているので
上位1%・3%・5%の成績も同時に計算できる。これは評価のためであって、
成績を見てから運用の設定を変えるためではない。

判定について先に:
  上位2%は1日あたり10点程度。的中率は6%前後の想定なので、
  1週分の期待的中は1本強。的中ゼロで終わる確率が3割近くある。
  数十点の段階では、どんな数字が出ても統計的な意味はない。
  ここで見るのは「パイプラインが正しく動いたか」だけ。

結果の取得について:
  --fetch を付けるとDBから確定成績を書き出してから照合する。
  以前は毎回このSQLを手で打っていたが、手順として忘れやすいので
  スクリプトに埋め込んだ。data_kubun='7' が確定成績を意味する。

腕別の集計:
  A / D / E' / E'' の4腕を並行記録しているので、pick_* 列があれば
  腕ごとの成績も出す。ただし1週あたり数点しかないので、
  数字を読むのは点数が数千に達してからである。

使い方:
  python evaluate.py
  python evaluate.py --fetch                 # DBから結果を取得してから照合
  python evaluate.py --fetch --days 7        # 直近7日分を取得
  python evaluate.py --out results_summary.csv
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

FRACS = [0.01, 0.02, 0.03, 0.05]
PAYOUT = 0.79

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="**/predictions_*.csv")
ap.add_argument("--results", default="**/results_*.csv")
ap.add_argument("--out", default=None)
ap.add_argument("--fetch", action="store_true",
                help="DBから確定成績を書き出してから照合する")
ap.add_argument("--days", type=int, default=14,
                help="--fetch で取得する期間(日)")
ap.add_argument("--pghost", default=os.environ.get("PGHOST", "192.168.10.2"))
args = ap.parse_args()


def fetch_results(days, host):
    """確定成績をDBから results_YYYYMMDD.csv へ書き出す。

    data_kubun='7' が確定成績。kakutei_chakujun や tansho_odds は
    text型で '00'/'0000' が未確定を意味するため、数値化の前に弾く。
    """
    end = date.today()
    start = end - timedelta(days=days)
    out = f"results_{end:%Y%m%d}.csv"
    sql = (
        "\\copy (SELECT (kaisai_nen||kaisai_gappi)::date AS race_date,"
        " race_code, umaban::int AS umaban, bamei,"
        " CASE WHEN kakutei_chakujun ~ '^[0-9]+$'"
        " AND kakutei_chakujun <> '00' THEN kakutei_chakujun::int END"
        " AS chaku,"
        " CASE WHEN tansho_odds ~ '^[0-9]+$' AND tansho_odds <> '0000'"
        " THEN CAST(tansho_odds AS NUMERIC)/10 END AS odds_fin"
        " FROM umagoto_race_joho"
        " WHERE data_kubun='7' AND keibajo_code BETWEEN '01' AND '10'"
        f" AND (kaisai_nen||kaisai_gappi) BETWEEN '{start:%Y%m%d}'"
        f" AND '{end:%Y%m%d}') TO '{out}' CSV HEADER")
    cmd = ["psql", "-h", host, "-U", "postgres", "-d", "postgres",
           "-c", sql]
    print(f"確定成績を取得中 ({start:%Y-%m-%d} 〜 {end:%Y-%m-%d}) → {out}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"取得に失敗しました:\n{r.stderr.strip()}")
    print(f"  {r.stdout.strip()}")


if args.fetch:
    fetch_results(args.days, args.pghost)

# 週ごとのサブフォルダに分かれているので再帰的に探す
pred_files = sorted(glob.glob(args.pred, recursive=True))
res_files = sorted(glob.glob(args.results, recursive=True))
if not pred_files or not res_files:
    sys.exit("predictions_*.csv または results_*.csv が見つかりません")
print(f"予想ファイル {len(pred_files)} 件 / 結果ファイル {len(res_files)} 件")

res = pd.concat([pd.read_csv(f) for f in res_files], ignore_index=True)
res["race_date"] = pd.to_datetime(res.race_date).dt.date
res["chaku"] = pd.to_numeric(res.chaku, errors="coerce")
res["odds_fin"] = pd.to_numeric(res.odds_fin, errors="coerce")
# 結合キー: race_code+umaban があればそれを使う。
# (race_date, bamei) だと同じ日に同名の馬が別レースにいた場合に取り違える。
KEY = (["race_code", "umaban"]
       if {"race_code", "umaban"} <= set(res.columns)
       else ["race_date", "bamei"])
if "umaban" in res.columns:
    res["umaban"] = pd.to_numeric(res.umaban, errors="coerce")
res = res.drop_duplicates(KEY)
print(f"結合キー: {KEY}")

# 各予想ファイルごとに上位x%を選び、それを積み上げる(実運用と同じ手順)
sel_all = {f: [] for f in FRACS}
allp = []
for pf in pred_files:
    p = pd.read_csv(pf)
    p["race_date"] = pd.to_datetime(p.race_date).dt.date
    p["batch"] = pf
    if "umaban" in p.columns:
        p["umaban"] = pd.to_numeric(p.umaban, errors="coerce")
    m = p.merge(res[KEY + ["chaku", "odds_fin"]], on=KEY, how="left")
    miss = m.chaku.isna().sum()
    print(f"  {pf}: {len(m)} 頭 / 結果と結合できず {miss} 頭")
    allp.append(m)
    for fr in FRACS:
        n = max(int(len(m) * fr), 1)
        sel_all[fr].append(m.nlargest(n, "ev"))

allp = pd.concat(allp, ignore_index=True)


def summarize(d, label):
    d = d[d.chaku.notna() & d.odds_fin.notna()]
    if len(d) == 0:
        return None
    win = (d.chaku == 1)
    ret = np.where(win, d.odds_fin, 0.0)
    obs, exp = win.sum(), d.p_mkt.sum()
    # 回収率の不確かさ(ブートストラップ)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ret), (3000, len(ret)))
    boot = ret[idx].mean(axis=1) * 100
    return dict(区分=label, 点数=len(d), 的中=int(obs),
                的中率=round(win.mean() * 100, 1),
                回収率=round(ret.mean() * 100, 1),
                回収率90percent区間=f"[{np.percentile(boot,5):.0f}, "
                                    f"{np.percentile(boot,95):.0f}]",
                市場の期待的中=round(exp, 2),
                比=round(obs / exp, 3) if exp > 0 else None,
                中央オッズ=round(d.odds_fin.median(), 1))


rows = []
for fr in FRACS:
    d = pd.concat(sel_all[fr], ignore_index=True)
    r = summarize(d, f"上位{fr*100:g}%")
    if r:
        rows.append(r)
rows.append(summarize(allp, "全馬(参考)"))
out = pd.DataFrame([r for r in rows if r])
print("\n=== 成績 ===")
print(out.to_string(index=False))

print("\n=== 日別(上位2%) ===")
d2 = pd.concat(sel_all[0.02], ignore_index=True)
for day, g in d2.groupby("race_date"):
    r = summarize(g, str(day))
    if r:
        print(f"  {day}: {r['点数']}点 / 的中 {r['的中']} / "
              f"回収率 {r['回収率']}%")

# ---- 腕別(pick_* 列がある予想ファイルのみ) --------------------------
ARMS = [("pick_a", "A  全レース"), ("pick_d", "D  除外"),
        ("pick_e", "E' 除外+20倍"), ("pick_e2", "E''除外+25倍")]
have = [(c, lab) for c, lab in ARMS if c in allp.columns]
if have:
    arm_rows = []
    for c, lab in have:
        sub = allp[allp[c].astype(str).str.lower() == "true"]
        r = summarize(sub, lab)
        if r:
            v = sub[sub.odds_fin.notna()]
            r["変化率中央"] = (f"{(v.odds_fin / v.odds - 1).median()*100:+.0f}%"
                            if len(v) else None)
            arm_rows.append(r)
    if arm_rows:
        print("\n=== 腕別 ===")
        print(pd.DataFrame(arm_rows).to_string(index=False))
        print("  1週あたり数点しかない。点数が数千に達するまで数字は読まない")
        print("  変化率中央 = 生成時オッズから確定オッズへの変化の中央値")

print("\n=== 買い目の明細(上位2%) ===")
cols = [c for c in ["race_date", "場", "race_bango", "umaban", "bamei",
                    "odds", "odds_fin", "chaku", "ev"] if c in d2.columns]
print(d2.sort_values(["race_date", "race_bango"])[cols].to_string(index=False))

# 生成時オッズと確定オッズのずれ(締切間際の変動を実地で確認する)
if "odds" in d2.columns:
    v = d2[d2.odds_fin.notna()]
    drift = (v.odds_fin / v.odds - 1) * 100
    print(f"\n=== 生成時オッズ → 確定オッズの変化(買い目のみ) ===")
    print(f"  中央値 {drift.median():+.1f}% / 平均 {drift.mean():+.1f}% / "
          f"範囲 {drift.min():+.0f}% 〜 {drift.max():+.0f}%")
    print("  検証では、前日から確定にかけて人気馬は短くなり人気薄は伸びる傾向")

if args.out:
    out.to_csv(args.out, index=False)
    d2.to_csv(args.out.replace(".csv", "_picks.csv"), index=False)
    if have and arm_rows:
        pd.DataFrame(arm_rows).to_csv(
            args.out.replace(".csv", "_arms.csv"), index=False)
    print(f"\n{args.out} に保存")

print("\n  注意: この点数では的中ゼロも連続的中も普通に起こる。"
      "統計的な評価には数千点規模の蓄積が必要。")
