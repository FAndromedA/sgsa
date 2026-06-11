/mnt/jfzn/models/Qwen3-4B-Thinking-2507 \
    /mnt/jfzn/models/Qwen3-4B-Instruct-2507 \
    /mnt/jfzn/models/Qwen3-4B-Base \

CUDA_VISIBLE_DEVICES=4 /mnt/jfzn/miniconda3/envs/zjh_dev/bin/python \
  experiments/analyze_qwen_attention_topk_reuse_exclude_edges.py \
  --model-paths /mnt/jfzn/models/Qwen3-4B-Base \
    /mnt/jfzn/models/Qwen3-4B-Thinking-2507 \
    /mnt/jfzn/models/Qwen3-4B-Instruct-2507 \
  --prompts-file experiments/attention_reuse_long_prompts.jsonl \
  --max-samples 1 \
  --max-seq-len 4096 \
  --k-values 64,128,256 \
  --layer-gaps 1,2,4 \
  --query-selection all \
  --score-chunk-size 16 \
  --exclude-sink-count 64 \
  --exclude-recent-count 64 \
  --aggregate-heads \
  --output outputs/attention_topk_reuse_exclude_edges.jsonl

/mnt/jfzn/miniconda3/envs/zjh_dev/bin/python \
experiments/summarize_attention_topk_reuse.py \
--input outputs/attention_topk_reuse_exclude_edges.jsonl \
--output outputs/attention_topk_reuse_exclude_edges_summary.csv

python experiments/plot_attention_topk_reuse_results.py \
  --summary outputs/attention_topk_reuse_exclude_edges_summary.csv \
  --jsonl outputs/attention_topk_reuse_exclude_edges.jsonl \
  --output-dir outputs/attention_topk_reuse_exclude_edges_figures \
  --report outputs/attention_topk_reuse_exclude_edges_plot_analysis.md