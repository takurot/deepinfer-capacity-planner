# 機能仕様書: deepinfer-capacity-planner

- 版: 0.1（ドラフト）
- 対象: DeepInfer `main` @ `5537415`、モデルは DeepSeek-V4.1-Flash 系
- 以下で `file:line` と書く箇所は、すべて DeepInfer リポジトリ内のパスを指す。

## 1. 目的

DeepInfer で推論をデプロイする運用者が、GPU を確保して起動する前に、次の問いに数値で答えられるようにする。

- 「この HBM 容量と並列構成、コンテキスト長で、KV cache は何ブロック確保でき、長文リクエストを同時に何本さばけるか」

現状、この値は GPU 上で起動して `profile_num_available_blocks`（`deepinfer/worker/base.py:211-268`）のログを読むまでわからない。手がかりは `doc/inference_deployment_qa.md` の経験則しかない。

## 2. スコープ

### 対象

- KV cache・state cache のサイズを、本体と同じ関数で厳密に計算する。
- 重みメモリ、活性化メモリ、CUDA Graph メモリを近似で見積もる。
- 確保できるブロック数と同時実行数を、幅を持たせて推定する。
- 複数構成を並べて比較する。

### 対象外

- 活性化メモリの厳密な予測。本体が実際にダミーの forward を流して測っており（`deepinfer/worker/model_runner.py:444-511`）、閉じた式は存在しない。
- Ascend（NPU）固有のメモリモデル。v0.1 では入力として受け付けるが、近似係数は NVIDIA の値を流用し、その旨を警告する。
- DeepSeek-V4.1 系以外のアーキテクチャ。

## 3. 用語

| 用語 | 意味 |
|---|---|
| block | KV cache の割り当て単位。`block_size` は、1 ブロックが圧縮前に何トークン分を表すか |
| state slot | V4.1 の state cache の 1 シーケンス分。数は `int(max_num_seqs * ratio) + 1` |
| 厳密値 | DeepInfer 本体と同じ関数を同じ入力で呼んで得た値 |
| 近似値 | 本ツール独自の推定式で得た値。必ず幅（low / high）で示す |

## 4. 入力

| 引数 | 必須 | 既定 | 説明 |
|---|---|---|---|
| `--model PATH` | ✓ | — | HF モデルのディレクトリ。`config.json` だけを読み、重みは読まない |
| `--ep`, `--dp`, `--cp` | ✓ / — / — | — / `=ep` / 1 | 並列設定。`cp` は `ep` を割り切れること |
| `--context-length` | ✓ | — | `max_model_len` に相当 |
| `--max-num-seqs` | — | 本体の既定値 | 同時シーケンスの上限 |
| `--block-size` | — | 本体の既定値 | |
| `--num-extra-experts` | — | 0 | `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` に相当。下限値として扱う |
| `--hbm-gb` | ✓ | — | GPU 1 枚あたりの HBM 容量 |
| `--gpu-memory-utilization` | — | 0.9 | `LLM(...)` の既定値に合わせる |
| `--state-slots-ratio` | — | 1.2 | `DEEPINFER_NUM_STATE_SLOTS_RATIO` |
| `--calibration FILE` | — | — | 実機ログから得た補正値（6.4 節） |
| `--format` | — | `table` | `table` / `json` / `markdown` |
| `--compare FILE` | — | — | 複数構成を YAML で列挙し、横並びで比較する |

入力の検証（起動前に失敗させる）:

- `config.json` が存在しない、またはパースできない → 終了コード 2。
- `ep % cp != 0` → 終了コード 2。
- `gpu_memory_utilization` が (0, 1] の範囲外 → 終了コード 2。

## 5. 出力

### 5.1 テーブル出力（既定）

```
[exact]  num_extra_experts (effective): 32 → 64   (rounded up so that routed+extra is divisible by ep=64)
[exact]  KV cache block: 12.4 KiB/block (block_size=64)
[exact]  state cache: 3.1 MiB/slot × 308 slots = 954 MiB
[approx] weights/rank: 21.3 GiB
[approx] activation+graph+workspace: 4.1 – 7.9 GiB
[approx] usable KV blocks: 180,000 – 210,000
[approx] concurrent sequences @131072 tokens: 11 – 13
WARNING: approx values are not measured. Calibrate with --calibration from a real run.
```

この出力例の数値は書式を示すためのもので、実際の計算結果ではない。

### 5.2 JSON 出力

項目ごとに `{"value": ..., "low": ..., "high": ..., "kind": "exact"|"approx", "source": "<根拠>"}` の形で出す。`exact` の項目は `low == high == value` とする。

## 6. 機能要件

### FR-1 モデル設定の読み込み

`deepinfer.model_config_loader.load_model_config`（`deepinfer/model_config_loader.py:8-23`）を呼んで `cfg` を得る。独自のパーサは書かない。DeepInfer 側でパースの仕様が変わっても追従できるようにするため。

### FR-2 冗長エキスパート数の確定

`deepinfer/model_executor/eplb/__init__.py:21-28` の `resolve_num_extra_experts` を呼ぶ。この関数は、`routed + extra` が `ep` で割り切れるように切り上げる。backbone と DFlash は本体と同様に別々に丸める。

### FR-3 KV cache / state cache の厳密計算

- `CacheEngine.get_cache_block_size(block_size, cfg)`（`deepinfer/worker/cache_engine.py:969-978`）を呼ぶ。
- `CacheEngine.get_state_shape(cfg, max_num_seqs)`（同 `:901-905`）を呼ぶ。
- どちらも内部では `_kvcache_layout(..., gpu_blocks=None, state_blocks=None)` を通る。この経路はテンソルを確保しない純粋な算術である（同 `:402-477`、`:581-652`）。
- 呼び出し回数を最小にする（docstring による要求）。比較モードでは、同じ `(cfg, block_size)` の結果をツール側でメモ化する。

### FR-4 state slot 数

`get_num_state_slots`（`cache_engine.py:202`）と同じ式 `int(max_num_seqs * ratio) + 1` で計算する。

### FR-5 重みメモリの近似

- `cfg` から次の量を求める。
  - routed experts: `(n_routed + extra) / ep`
  - shared experts、attention、norm: 各ランクに複製
  - embedding / lm_head: 実装に従う
- 量子化形式（FP8 / FP4）ごとのバイト数は、`deepinfer/model_executor/models/deepseek.py` と `v4_modules/` の実装から導く。
- 実装時に、シャーディングの式をコードから抽出して表にまとめる。その表を本仕様の付録 A として追記する（未作成）。

### FR-6 活性化メモリの近似

- 既定値は `[low, high]` の係数で与える。v0.1 の値は暫定で、初回の実機キャリブレーションで置き換える。
- 本体の `profile_run` は、MoE の負荷不均衡を `DEEPINFER_FORCE_IMBALANCE_RATIO=1.8` で模擬したうえで測っている（`model_runner.py:505-510`）。このため、近似は悲観側に寄せる。

### FR-7 確保可能ブロック数

本体の式（`worker/base.py:235-244`）をなぞる。

```
available = total_hbm * gpu_memory_utilization - peak_allocated - state_size * 2
num_gpu_blocks = available // block_bytes
```

- `peak_allocated` には、重みと活性化の low / high を入れる。結果も幅で出す。
- 同時実行数は `num_gpu_blocks // ceil(context_length / block_size)` とする。
- 圧縮 KV や SWA のために、シーケンス 1 本が消費するブロック数が線形にならない場合は、FR-3 の layout から 1 シーケンスあたりのブロック消費量を算出して使う。

### FR-8 キャリブレーション

- 実機ログから `peak_allocated_memory` と `num_gpu_blocks`（`worker/base.py:250-262` の `logger.info`）を貼り付けた YAML を受け取る。
- そこから活性化係数を逆算して保存し、次回以降の近似に使う。
- 補正値と入力構成の差が大きいときは、警告を出す。対象は `context_length` と `max_num_seqs` が 2 倍以上違う場合。

### FR-9 比較モード

YAML に複数の構成を書くと、同じ `cfg` で各構成を計算し、項目を行、構成を列とした表を出す。

## 7. 非機能要件

- GPU も CUDA も不要。CPU 版の torch と DeepInfer を import できる環境で動くこと。
  - **未検証:** DeepInfer の `_C` 拡張は CUDA 前提でビルドされる。FR-3 の経路が `_C` を import せずに済むかは、実装の最初の作業で確かめる。
  - import できない場合は、`_kvcache_layout` の算術をツール側に移植し、本体と一致することを CI で検証する方式に切り替える。
- 1 構成の計算は 5 秒以内（重みは読まない）。
- 近似値はすべて `approx` と表示し、黙って厳密値のように見せない。

## 8. 受け入れ基準

1. 同じ `config.json` に対し、FR-3 で得た KV block のバイト数が、実機ログ（`cache_engine.py:935,945` の `kb_per_block`）と一致する。GPU 環境で 1 回確かめれば、以後は固定値の回帰テストにする。
2. FR-2 の結果が、本体で実際に使われた値（`benchmark.py` の `result_jsonl` にある `num_extra_experts`）と一致する。
3. キャリブレーションした構成で、`num_gpu_blocks` の実測値が [low, high] の範囲に入る。
4. 不正な入力（4 章）が、エンジンを起動せずに終了コード 2 で失敗する。

## 9. 未決事項とリスク

- 重みのシャーディング式は、`deepseek.py` からの手作業での再現になるため、ずれる可能性がある。付録 A を作るときに、実機の `torch.cuda.memory_allocated()` と照合する。
- 活性化メモリは、構成によって大きく変わる可能性がある（推測）。固定係数で大きく外れる構成があれば、係数をパラメータ化する（`max_num_batched_tokens` に依存する項を分けるなど）。
- DFlash や YoCo を有効にしたときのメモリ増分は、v0.1 では扱わない。
