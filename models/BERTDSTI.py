from .BERTDSTMain import BERTDSTMain
from torch import nn
from models.statetrackingutils import SlotFillingIntentDetectionModelOutput
import torch


class IntentClassifier(nn.Module):
    def __init__(self, in_size, out_size, dropout_prob=0.5):
        super(IntentClassifier, self).__init__()
        self.classifier = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_keys = None

    def forward(self, token_out):
        token_out = self.dropout(token_out)
        return self.classifier(token_out)


class Pooler(nn.Module):
    def __init__(self, pooler_index, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pooler_index = pooler_index

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.pooler_index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTDSTI(BERTDSTMain):
    def __init__(self, checkpoint_name, tokenizer, *args, **kwargs):
        self.intent_keys = None
        self.intent_pooler_layer = None
        self.intent_classifier = None
        self.eval_to_return = []
        super(BERTDSTI, self).__init__(checkpoint_name, tokenizer, args, kwargs)

    def post_init(self, info, *args, **kwargs):
        super().post_init(info)
        self.intent_keys = [item.lower() for item in info["all_intents"]]
        self.intent_keys.sort()
        self.intent_pooler_layer = Pooler(
            pooler_index=1, hidden_size=self.config.hidden_size
        )
        self.intent_classifier = IntentClassifier(
            self.config.hidden_size, len(info["all_intents"])
        )

        print(
            f"Loaded BERT-DSTINTENT model with intents: {self.intent_keys}", flush=True
        )

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
            intent = self.intent_pooler_layer(all_tokens_out)
            intent = self.intent_classifier(intent)
            # TODO always make sure that the intent detection token is token 1 of the sequence
            loss = self.compute_loss(
                (model_out, gate_logits, span_logits, intent), extra_info, accel
            )
            return loss, (model_out, gate_logits, span_logits, intent)
        elif get_output_for_metrics:
            self.get_output_for_metrics(model_out, input_info, extra_info)
        else:
            return None

    def compute_loss(self, model_out, ground_truth, accel, *args, **kwargs):
        intent_loss_percentage = 0.4
        slot_filling_loss_percentage = 0.6

        slot_filling_loss = super().compute_loss(
            model_out, ground_truth, accel, args, kwargs
        )
        loss_fn = nn.CrossEntropyLoss()
        intent_logits = model_out[3]
        intent_gt = [
            self.intent_keys.index(item.intent.lower()) for item in ground_truth
        ]
        intent_loss = loss_fn(
            intent_logits,
            accel.to_device_single(accel.prepare_data(torch.tensor(intent_gt))),
        )

        return (
            intent_loss_percentage * intent_loss
            + slot_filling_loss_percentage * slot_filling_loss
        )

    def get_output_for_metrics(self, model_out, items, extra_info, *args, **kwargs):
        outs = super(BERTDSTI, self).get_output_for_metrics(
            model_out, items, len(extra_info), extra_info, should_return=True
        )
        for bs_index in range(len(extra_info)):
            outs[bs_index].intents = []
            outs[bs_index].set_intent(
                self.intent_keys[torch.max(model_out[3][bs_index], dim=0).indices]
            )
            self.eval_to_return.append(outs[bs_index])

    def get_human_readable_output(self, input_info, tokens, *args, **kwargs):
        model_out, gate_logits, span_logits = super().forward(input_info, args, kwargs)
        all_tokens_out = model_out.last_hidden_state  # out
        intent = self.intent_pooler_layer(all_tokens_out)
        intent = self.intent_classifier(intent)

        out = SlotFillingIntentDetectionModelOutput(
            dialogue="0",
            utterance="0",
            tokens=tokens,
        )
        out.set_intent(self.intent_keys[torch.max(intent[0], dim=0).indices])

        for slot_key, gate_out_logits in gate_logits.items():
            span_out_logits = span_logits[slot_key]
            gate_out = torch.max(gate_out_logits[0], dim=0).indices.item()
            if gate_out == 1:
                out.set_slot(slot_key, "dontcare")
            elif gate_out == 2:
                span_out = torch.max(span_out_logits[0], dim=0).indices
                out.set_slot(
                    slot_key,
                    self.tokenizer.convert_tokens_to_string(
                        tokens[span_out[0] : span_out[1] + 1]
                    ),
                )

        return out
