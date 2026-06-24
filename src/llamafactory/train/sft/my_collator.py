from ...data import SFTDataCollatorWith4DAttentionMask
from ...extras.constants import IGNORE_INDEX
from torch.nn.utils.rnn import pad_sequence
import torch

class ThinkAnswerCollator(SFTDataCollatorWith4DAttentionMask):
    """
    Adds segment-specific labels on top of the original SFT collator output:
        think_labels : (B, L), supervises the <THINK> segment
        ans_labels   : (B, L), supervises the <ANSWER> segment
    Other batch keys are preserved.
    """

    def __init__(self, *args, **kwargs):
        self.id_think = kwargs.pop("id_think")
        self.id_ans   = kwargs.pop("id_ans")
        super().__init__(*args, **kwargs)

        self.ignore_index = IGNORE_INDEX

    def __call__(self, features):
        batch = super().__call__(features)

        input_ids = batch["input_ids"]
        think_lbl, ans_lbl = [], []

        for sample_idx, ids in enumerate(input_ids):  # ids: Tensor (L,)
            think_pos = (ids == self.id_think).nonzero(as_tuple=False)
            ans_pos = (ids == self.id_ans).nonzero(as_tuple=False)
            if think_pos.numel() == 0 or ans_pos.numel() == 0:
                raise ValueError(
                    f"Missing <THINK> or <ANSWER> token after preprocessing in batch sample {sample_idx}. "
                    "Please check the output template and cutoff_len."
                )

            p_think = think_pos[0, 0]
            p_ans = ans_pos[0, 0]
            if p_think >= p_ans:
                raise ValueError(
                    f"Invalid token order in batch sample {sample_idx}: <THINK> must appear before <ANSWER>."
                )

            lbl_think = ids.clone()
            lbl_think[:p_think + 1] = self.ignore_index
            lbl_think[p_ans:] = self.ignore_index
            think_lbl.append(lbl_think)

            lbl_ans = ids.clone()
            lbl_ans[:p_ans + 1] = self.ignore_index
            ans_lbl.append(lbl_ans)

        device = input_ids.device
        batch["think_labels"] = pad_sequence(think_lbl, batch_first=True,
                                             padding_value=self.ignore_index).to(device)
        batch["ans_labels"]   = pad_sequence(ans_lbl,   batch_first=True,
                                             padding_value=self.ignore_index).to(device)
        return batch
