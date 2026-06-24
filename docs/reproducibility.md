# Reproducibility Notes

## Dataset Splits

The dataset contains 2,492 image-text samples and is split into train, validation, and test subsets.

| Split | Samples |
| --- | ---: |
| Train | 1,993 |
| Validation | 249 |
| Test | 250 |

## Overall Comfort Level Distribution

| Split | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 416 | 341 | 787 | 216 | 233 |
| Validation | 53 | 42 | 97 | 28 | 29 |
| Test | 43 | 49 | 102 | 26 | 30 |

## Dimension-Level Score Distribution

### Architectural Configuration

| Split | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 467 | 303 | 559 | 431 | 233 |
| Validation | 58 | 37 | 68 | 57 | 29 |
| Test | 52 | 42 | 70 | 56 | 30 |

### Color Scheme

| Split | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 417 | 349 | 776 | 218 | 233 |
| Validation | 53 | 43 | 96 | 28 | 29 |
| Test | 43 | 51 | 99 | 27 | 30 |

### Lighting Ambience

| Split | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 416 | 503 | 606 | 235 | 233 |
| Validation | 54 | 64 | 73 | 29 | 29 |
| Test | 43 | 71 | 76 | 30 | 30 |

## Composite Loss

The modified SFT trainer separates the assistant output into:

- explanation segment: after `<THINK>` and before `<ANSWER>`
- answer segment: final comfort rating after `<ANSWER>`

The released code uses the final recommended paper setting:

```text
L = 0.4 * L_think + 0.6 * L_answer
```

`L_answer` is computed only on the final numeric comfort level token (`1`-`5`) to match the classification target used for the overall comfort level.

## Base Training Setup

- Framework: LLaMA-Factory
- Base model: Qwen2.5-VL-7B-Instruct
- Fine-tuning: LoRA
- Template: qwen2_vl
- Learning rate: 5e-5
- Scheduler: cosine
- Batch size: 2
- Gradient accumulation steps: 8
- Epochs: 10
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0
