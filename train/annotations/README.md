# Annotations / 标注说明

This folder stores human-approved song annotations for YesTiger.

本目录保存 YesTiger 的人工确认歌曲标注。

## Current Schema / 当前 Schema

Current annotation files use `annotation_version: 0.2.0`.

当前标注文件使用 `annotation_version: 0.2.0`。

```json
{
  "annotation_version": "0.2.0",
  "song": {
    "song_id": "mayoiuta",
    "title": "迷星叫",
    "artist": "MyGO!!!!!",
    "franchise": "bang_dream",
    "audio_path": "songs/迷星叫.mp3",
    "bpm": 95,
    "call_bpm": 95,
    "call_bar_multiplier": 1.0,
    "meter": "4/4"
  },
  "segments": [],
  "call_spans": []
}
```

## `segments` / 音乐结构层

`segments` should describe music structure only.

`segments` 只描述音乐结构。

```json
{
  "start": 36.53,
  "end": 41.58,
  "music_label": "pre_chorus",
  "notes": ""
}
```

`music_label` is the fine-grained human annotation and the single source of
truth. The training pipeline derives a coarse `music_family` automatically.

`music_label` 是人工填写的细粒度标签，也是唯一真值。训练程序自动推导粗粒度
`music_family`，不要求所有标注重复填写父标签。

Hierarchy / 两层结构：

| `music_family` | `music_label` |
|---|---|
| `intro` | `intro` |
| `verse` | `verse` |
| `pre_chorus` | `pre_chorus`, `pre_chorus_build` |
| `chorus` | `chorus`, `post_chorus` |
| `instrumental` | `instrumental_break`, `solo` |
| `bridge` | `bridge` |
| `outro` | `outro` |

`music_family` may be materialized as an optional field for interchange. If it
is present, the validator requires it to match `music_label`.

交换文件可以显式增加可选的 `music_family` 字段；如果填写，验证器会检查父子
标签是否一致。

Fine-grained trainable structure labels:

参与训练的细粒度结构标签：

```text
intro
verse
pre_chorus
pre_chorus_build
chorus
post_chorus
instrumental_break
bridge
solo
outro
```

Annotation-only marker:

仅供人工标注使用的辅助标记：

```text
end
```

`end` records where the annotator considers the song finished. Keep it in the
annotation with:

`end` 用于提醒标注者歌曲在何处结束。请在该段保留：

```json
{
  "music_label": "end",
  "notes": "annotation_marker_only; excluded_from_training_and_inference"
}
```

It is excluded from model labels, loss, evaluation metrics, and prediction
exports.

它不参与模型分类、loss、评价指标或预测结果导出。

## `call_spans` / Call 层

`call_spans` is an optional downstream planning layer. Structure training does
not require it. Keep an empty array when calls have not been assigned yet:

`call_spans` 是可选的下游应援规划层，不参与当前结构模型训练。尚未分配 call
时保留空数组即可：

```json
"call_spans": []
```

Future LLM/RAG or rule-based planners may populate this array without changing
the manually approved structure labels.

后续可以由 LLM、RAG 或规则系统填充该数组，不需要修改已经人工确认的结构标签。

When present, `call_spans` should describe call/mix behavior only.

填写后，`call_spans` 只描述 call / mix 行为。

```json
{
  "start": 73.18,
  "end": 83.27,
  "call_role": "mix",
  "recommended_actions": ["bandor_mix"],
  "notes": ""
}
```

Supported roles:

支持的角色：

```text
keepspace
rhythmcall
mix
underground_gei
```

`recommended_actions` must use IDs from:

`recommended_actions` 必须来自：

```text
knowledge/call_mix_library.json
```

## Validation / 验证

Run:

运行：

```powershell
python scripts\validate_annotation.py
```

The validator checks:

验证器会检查：

- timeline order
- music labels
- call roles
- known action IDs
- action category vs. call role
- rough duration and downbeat alignment

- 时间顺序
- 音乐结构标签
- call 角色
- action ID 是否存在
- action 分类是否匹配 call role
- 粗略长度和 downbeat 对齐

## Dataset Export / 数据集导出

For training data:

导出训练数据：

```powershell
.\.venv\Scripts\python.exe scripts\build_pipeline_dataset.py --all
```

Output:

输出：

```text
datasets/pipeline/bar_rows.jsonl
datasets/pipeline/action_pairs.jsonl
datasets/pipeline/manifest.json
datasets/pipeline/songs/*.sequence.json
```
