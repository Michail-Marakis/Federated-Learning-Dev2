import numpy as np
import torch
from tqdm import tqdm

def get_flatten_features(model, data_loader, args):
    progress_bar_eval = tqdm(range(len(data_loader)))
    flatten_hidden_state_history = []

    with torch.inference_mode():
        for batch in data_loader:
            batch = {
                'input_ids': batch['input_ids'].cuda(),
                'labels': batch['labels'].cuda(),
                'attention_mask': batch['attention_mask'].cuda()
            }

            # ✅ FIX: always return hidden states
            outputs = model(**batch, output_hidden_states=True)

            tmp_hidden_state = []
            for idx_layer in range(len(outputs.hidden_states)):
                hidden_state_cur_layer = torch.squeeze(outputs.hidden_states[idx_layer])

                if args.feature_token == 'avg':
                    tmp_hidden_state.append(
                        torch.mean(hidden_state_cur_layer, dim=0).cpu().numpy()
                    )
                else:
                    tmp_hidden_state.append(
                        hidden_state_cur_layer[-1].cpu().numpy()
                    )

            flatten_hidden_state_history.append(
                np.array(tmp_hidden_state, dtype=np.float64)
                .reshape((len(tmp_hidden_state) * len(tmp_hidden_state[0])))
            )

            progress_bar_eval.update(1)
            progress_bar_eval.set_description("extracting feature")

    return flatten_hidden_state_history
