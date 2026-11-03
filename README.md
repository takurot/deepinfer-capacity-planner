# deepinfer-capacity-planner

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) のデプロイ構成を、GPU を確保する前に見積もるサイジング用 CLI。

> **ステータス: 仕様策定段階（未実装）。** 機能仕様は [docs/SPEC.md](docs/SPEC.md) を参照。

## 何がわかるか

HF の `config.json`、並列設定（EP / DP / CP）、コンテキスト長、GPU の HBM 容量を入力すると、次の値を出す。

| 項目 | 精度 | 根拠 |
|---|---|---|
| KV cache 1 ブロックのバイト数 | **厳密** | DeepInfer の `CacheEngine.get_cache_block_size` をそのまま呼ぶ |
| state cache のサイズと state slot 数 | **厳密** | `CacheEngine.get_state_shape` と `DEEPINFER_NUM_STATE_SLOTS_RATIO` |
| 実効の冗長エキスパート数（EP 割り切りへの切り上げ） | **厳密** | `resolve_num_extra_experts` |
| ランクあたりの重みメモリ | 近似 | config から算出する |
| 活性化 / CUDA Graph / ワークスペース | 近似（幅で表示） | 本体では `profile_run` の実測に依存するため |
| 確保できる KV ブロック数と同時実行リクエスト数 | 近似（楽観〜悲観の幅） | 上記の値を組み合わせる |

近似値は必ず「近似」と明示し、実機ログ（`profile_num_available_blocks`）で補正するための入力を受け付ける。

## 使い方（予定）

```bash
deepinfer-capacity-planner \
  --model /path/to/DeepSeek-V4.1-Flash \
  --ep 64 --dp 64 --cp 1 \
  --context-length 131072 --max-num-seqs 256 \
  --num-extra-experts 32 \
  --hbm-gb 192 --gpu-memory-utilization 0.9
```

重みファイル本体は不要で、`config.json` だけで動く。

## 前提

- 厳密計算の部分は、DeepInfer パッケージ（CPU 版の torch で import できること）に依存する。
- DeepInfer の対象コミットは `5537415`（2026-11 時点の `main`）。

## ライセンス

[MIT](LICENSE)
