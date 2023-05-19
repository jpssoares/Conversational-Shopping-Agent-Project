import torch
from torch import nn
import transformers
from transformers import AutoModel, AutoConfig
import numpy as np
import math

from source.statetrackingutils import SlotFillingIntentDetectionModelOutput


class CLSClassifier(nn.Module):
    def __init__(self, in_size, dropout_prob=.5, out_size=3):
        super(CLSClassifier, self).__init__()
        self.classifier = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, cls_out):
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out)


class SpanClassifier(nn.Module):
    def __init__(self, in_size, dropout_prob=.5, out_size=2):
        super(SpanClassifier, self).__init__()
        self.classifier = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, span_out):
        span_out = self.dropout(span_out)
        return self.classifier(span_out)


class BERTDSTMain(nn.Module):
    def __init__(self, checkpoint_name, tokenizer, *args, **kwargs):
        super(BERTDSTMain, self).__init__()
        self.model = AutoModel.from_pretrained(checkpoint_name)
        self.config = AutoConfig.from_pretrained(checkpoint_name)
        self.tokenizer = tokenizer
        self.checkpoint_name = checkpoint_name

    def post_init(self, info, *args, **kwargs):
        self.slot_keys = info['all_slots']
        self.slot_keys.sort()

        self.span_classifiers = nn.ModuleDict()
        self.slot_gates = nn.ModuleDict()

        for slot_key in self.slot_keys:
            slot_key_repr = slot_key.replace('.', '__')  # MODULE NAMES CANNOT CONTAIN DOTS
            self.slot_gates[slot_key_repr] = CLSClassifier(self.config.hidden_size)
            self.span_classifiers[slot_key_repr] = SpanClassifier(self.config.hidden_size)
            # we can also do self.model.get_input_embeddings().embedding_dim

        print("Loaded BERT-DST model", flush=True)

    def forward(self, input_info=None, *args, **kwargs):
        if 'label' in input_info:
            input_info.pop('label')

        model_out = self.model(**input_info)
        cls_out = getattr(model_out, "pooler_output", model_out.last_hidden_state[:, 0, :])
        all_out = model_out.last_hidden_state
        gate_logits, span_logits = dict(), dict()

        for slot_key in self.slot_keys:
            slot_key_repr = slot_key.replace('.', '__')
            gate_logits[slot_key] = self.slot_gates[slot_key_repr](cls_out)
            span_logits[slot_key] = self.span_classifiers[slot_key_repr](all_out)

        return model_out, gate_logits, span_logits

    def get_base_model(self):
        return self.model

    def compute_loss(self, model_out, ground_truth, accel, *args, **kwargs):
        gate_loss_percent, span_loss_percent = .8, .2
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        for slot_key in self.slot_keys:
            if slot_key not in ground_truth[0].slots:
                # print(f'Slot {slot_key} not found in in gt! Skipping...')
                continue
            gt_by_slot = [item.slots[slot_key] for item in ground_truth]
            gate_labels_by_slot, span_labels_by_slot = self._get_ground_truth_labels_of_slot(gt_by_slot, accel)
            logit_gate, logit_span = model_out[1][slot_key], model_out[2][slot_key]
            logit_span = logit_span.to('cpu')
            span_labels_by_slot = span_labels_by_slot.to('cpu')
            loss_span_start = loss_fn(logit_span[:, :, 0], span_labels_by_slot[:, 0])
            loss_span_end = loss_fn(logit_span[:, :, 1], span_labels_by_slot[:, 1])
            if math.isnan(loss_span_end):
                loss_span_start = 0
                loss_span_end = 0
            loss_gate = loss_fn(logit_gate, torch.max(gate_labels_by_slot, dim=1)[1])
            total_loss += loss_span_start * span_loss_percent / 2 + loss_span_end * span_loss_percent / 2 + loss_gate * gate_loss_percent
        return total_loss

    def _get_ground_truth_labels_of_slot(self, gt_by_slot, accel):
        gate_labels_by_slot = np.array([[0, 0, 0] for _ in range(len(gt_by_slot))])
        span_labels_by_slot = np.array([[item.span_start, item.span_end] for item in gt_by_slot])
        # TODO check if you need an array at all
        for i in range(len(gate_labels_by_slot)):
            gate_labels_by_slot[i][gt_by_slot[i].slot_gate] = 1
        # set span label of non-present slots to -100
        for i in range(len(gate_labels_by_slot)):
            if gate_labels_by_slot[i][0] == 1 or gate_labels_by_slot[i][1] == 1:
                span_labels_by_slot[i] = np.array([-100, -100])
        return (accel.to_device_single(accel.prepare_data(torch.tensor(gate_labels_by_slot))),
                accel.to_device_single(accel.prepare_data(torch.tensor(span_labels_by_slot))))

    def get_output_for_metrics(
            self,
            model_out,
            items,
            bs_index,
            extra_info,
            should_return=False,
            *args,
            **kwargs
    ):
        outs = []
        for i in range(bs_index):
            tokens = extra_info[i].tokens
            out = SlotFillingIntentDetectionModelOutput(
                dialogue=extra_info[i].dialogue_id,
                utterance=extra_info[i].utterance_index,
                tokens=extra_info[i].tokens,
            )
            for slot_key, gate_out_logits in model_out[1].items():
                span_out_logits = model_out[2][slot_key]
                gate_out = torch.max(gate_out_logits[i], dim=0).indices.item()
                if gate_out == 1:
                    out.set_slot(slot_key, 'dontcare')
                elif gate_out == 2:
                    span_out = torch.max(span_out_logits[i], dim=0).indices
                    out.set_slot(slot_key, self.tokenizer.convert_tokens_to_string(tokens[span_out[0]:span_out[1] + 1]))
            outs.append(out)
        if should_return:
            return outs


    def get_output_for_metrics_force(self):
        return None

    def get_human_readable_output(self, model_out, tokens):
        out = self.get_output_for_metrics(
            model_out,
            [],
            0,
            {"tokens": tokens},
        )
        return out
