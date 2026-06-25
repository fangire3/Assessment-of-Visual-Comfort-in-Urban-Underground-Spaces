# Dataset

This folder contains the cleaned ShareGPT-style multimodal dataset splits:

| Split | Samples |
| --- | ---: |
| Train | 1,993 |
| Validation | 249 |
| Test | 250 |
| Total | 2,492 |

Each sample has:

- `messages`: system, user, and assistant messages
- `images`: a list containing one image file name

The assistant response follows:

```text
<THINK>
场景描述：...
建筑布局：...
<建筑布局评分：1-5>
色彩搭配：...
<色彩搭配评分：1-5>
照明设计：...
<照明设计评分：1-5>
<ANSWER>
整体舒适度等级：1-5。
```

## Image Files

Images are not included in this repository. To request the original image data, please contact zhoubiao@tongji.edu.cn.

Recommended placement after copying the dataset into LLaMA-Factory:

```text
LLaMA-Factory/data/images/
```

The current JSON files store image file names only, such as:

```json
"images": ["2038.jpg"]
```

If needed, update the JSON image entries to include `images/` as a prefix.
