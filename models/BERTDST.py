from .BERTDSTMain import BERTDSTMain
from torch import nn
from models.statetrackingutils import SlotFillingIntentDetectionModelOutput
import torch


class BERTDST(BERTDSTMain):
    def __init__(self, checkpoint_name, tokenizer, *args, **kwargs):
        self.eval_to_return = []
        super(BERTDST, self).__init__(checkpoint_name, tokenizer, args, kwargs)

    def post_init(self, info, *args, **kwargs):
        super().post_init(info)
        print(f"Loaded BERT-DST model", flush=True)

    def forward(
        self,
        input_info=None,
        extra_info=None,
        compute_loss=False,
        get_output_for_metrics=False,
        get_output_for_metrics_end=False,
        model_out=None,
        accel=None,
        *args,
        **kwargs,
    ):
        if get_output_for_metrics_end:
            e = self.eval_to_return
            self.eval_to_return = []
            return e
        elif extra_info is not None and compute_loss:
            model_out, gate_logits, span_logits = super().forward(
                input_info, args, kwargs
            )
            all_tokens_out = model_out.last_hidden_state  # out
            # TODO always make sure that the intent detection token is token 1 of the sequence
            loss = self.compute_loss(
                (model_out, gate_logits, span_logits), extra_info, accel
            )
            return loss, (model_out, gate_logits, span_logits)
        elif get_output_for_metrics:
            self.get_output_for_metrics(model_out, input_info, extra_info)
        else:
            return None

    def compute_loss(self, model_out, ground_truth, accel, *args, **kwargs):
        return super().compute_loss(model_out, ground_truth, accel, args, kwargs)

    def get_output_for_metrics(self, model_out, items, extra_info, *args, **kwargs):
        outs = super(BERTDST, self).get_output_for_metrics(
            model_out, items, len(extra_info), extra_info, should_return=True
        )
        for bs_index in range(len(extra_info)):
            self.eval_to_return.append(outs[bs_index])

    def get_human_readable_output(self, model_out, tokens):
        out = self.get_output_for_metrics(
            model_out,
            [],
            0,
            {"tokens": tokens},
        )
        return out
