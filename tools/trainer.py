import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from tqdm import tqdm
from phase_model_pkg.tools.label_utils import pd_collate_fn
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    def __init__(self, patience=5, mode='max', delta=0.001, save_path='best_model.pt', max_f1_gap=0.15):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.max_f1_gap = max_f1_gap

    def __call__(self, val_score, train_score, model, epoch=None):
        f1_gap = abs(val_score - train_score)

        if f1_gap > self.max_f1_gap:
            print(
                f"Skipped: val_f1 ({val_score:.4f}) - train_f1 ({train_score:.4f}) = {f1_gap:.4f} > {self.max_f1_gap}")
            self.counter += 1
        else:
            if self.best_score is None:
                self.best_score = val_score
                self.best_epoch = epoch
                self.save_checkpoint(model)
                self.counter = 0
                print(f'Initial model saved (epoch {epoch}) | val_f1={val_score:.4f}, train_f1={train_score:.4f}')
            elif ((self.mode == 'max' and val_score > self.best_score + self.delta) or
                  (self.mode == 'min' and val_score < self.best_score - self.delta)):
                print(f'New best model at epoch {epoch}: val_f1 improved ({self.best_score:.4f} → {val_score:.4f})')
                self.best_score = val_score
                self.best_epoch = epoch
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                print(f'No improvement: val_f1={val_score:.4f} | patience {self.counter}/{self.patience}')

        if self.counter >= self.patience:
            print(f'EarlyStopping triggered after {self.patience} rounds without valid improvement.')
            self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)


# 定义训练函数
def train_model(model, train_dataset, val_dataset, criterion, optimizer
                , num_epochs, device, early_stopping, y_key=0, sequence_level=False, counter_features=False
                , traditional_features=False):
    model.to(device)
    # scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        # print(all_labels)
        for batch in tqdm(train_dataset):
            optimizer.zero_grad()
            batch_episodes, masks = pd_collate_fn(batch)
            # batch_episodes = [episode.to(device) for episode in batch_episodes]
            masks = masks.to(device)
            sequence_targets = []
            frame_labels = []
            frame_masks = []
            for i, episode in enumerate(batch_episodes):
                episode_labels = torch.tensor([frame['global'].y[y_key] for frame in episode]).to(device)
                # print(episode_labels)
                if torch.tensor(4) in episode_labels or torch.tensor(5) in episode_labels:
                    episode_labels = (episode_labels == 5).long()
                if sequence_level:
                    sequence_target, _ = torch.mode(episode_labels)
                    sequence_targets.append(sequence_target)
                # print(episode_labels)
                frame_labels.append(episode_labels)
            if sequence_level:
                frame_predictions = model(batch_episodes, masks, y_key, device, sequence_level=True
                                          , counter_features=counter_features)
                frame_labels = torch.stack(sequence_targets).to(device)
                loss = criterion(frame_predictions, frame_labels, mode='sequence')
                # print(frame_labels, frame_labels.shape)
            else:
                frame_predictions = model(batch_episodes, masks, y_key, device
                                          , counter_features=counter_features
                                          , traditional_features=traditional_features)  # [batch_size, T, num_classes]
                # frame_labels = torch.stack(frame_labels).to(device)
                frame_labels = pad_sequence(frame_labels, batch_first=True).to(device)

            # print(frame_labels)
            loss = criterion(frame_predictions, frame_labels, mask=masks)
            # print(loss)
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if model.num_classes == 2:
                probs = torch.sigmoid(frame_predictions)  # [B]
                # print(probs)
                preds = (probs > 0.5).long().view(-1)
                # print(preds)
            else:
                preds = torch.argmax(frame_predictions, dim=-1).view(-1)
            preds = preds.view(-1)  # [B*T]
            labels = frame_labels.view(-1)  # [B*T]
            masks = masks.view(-1)  # [B*T]
            if sequence_level:
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
            else:
                # 只保留有效帧
                preds = preds[masks == 1].cpu().numpy()
                labels = labels[masks == 1].cpu().numpy()
                # print(preds.shape, labels.shape)
            all_preds.extend(preds)
            all_labels.extend(labels)

        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_f1_each = f1_score(all_labels, all_preds, average=None)
        train_matrix = confusion_matrix(all_labels, all_preds)
        print(train_matrix)
        # 计算验证集损失和 F1-score
        if sequence_level:
            val_loss, val_f1 = evaluate_model(model, val_dataset, criterion, device, y_key=y_key, sequence_level=True
                                              , counter_features=counter_features
                                              , traditional_features=traditional_features)
        else:
            val_loss, val_f1 = evaluate_model(model, val_dataset, criterion, device, y_key=y_key
                                              , counter_features=counter_features
                                              , traditional_features=traditional_features)

        # w = model.gnn_layers[0].msg_mlp[0].weight
        # w.register_hook(lambda g: print("grad finite (gnn msg_mlp[0].weight):", bool(torch.isfinite(g).all())))

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {total_loss / len(train_dataset):.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        print(train_f1_each)
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.mean().item()}")
        #     print(f"{name}: {param.grad}")
        #     break
        early_stopping(val_f1, train_f1, model, epoch=epoch)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        # torch.cuda.empty_cache()
    return train_f1, train_f1_each


def evaluate_model(model, val_dataset, criterion, device, y_key=0, sequence_level=False, counter_features=False
                   , traditional_features=False):
    model.eval()
    model.to(device)
    total_loss = 0.0
    all_preds, all_logits, all_labels = [], [], []

    with torch.no_grad():
        batch_episodes = val_dataset.intervals
        for episode in batch_episodes:
            episode_labels = torch.tensor([frame['global'].y[y_key] for frame in episode]).to(device)

            if torch.tensor(4) in episode_labels or torch.tensor(5) in episode_labels:
                episode_labels = (episode_labels == 5).long()
            if sequence_level:
                # print(episode_labels)
                # print(episode_labels.shape)
                sequence_target, _ = torch.mode(episode_labels)
                # print(sequence_target)
                logits, preds, transformer_out, _, _ = model.predict_episode(episode, device, sequence_level=True
                                                                             , counter_features=counter_features
                                                                             , traditional_features=traditional_features
                                                                             )
                all_preds.append(preds)
                all_labels.append(sequence_target)
                all_logits.append(logits)
            else:
                logits, preds, transformer_out, _, _ = model.predict_episode(episode, device
                                                                             , counter_features=counter_features
                                                                             , traditional_features=traditional_features
                                                                             )
                loss = criterion(logits, episode_labels)
                labels = episode_labels.view(-1).cpu().numpy()
                preds = preds.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                total_loss += loss.item()
    if sequence_level:
        all_preds = torch.stack(all_preds).to(device)
        all_labels = torch.stack(all_labels).to(device)
        all_logits = torch.stack(all_logits).to(device)
        total_loss = criterion(all_logits, all_labels, mode='sequence')
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        return total_loss, f1
    else:
        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_each = f1_score(all_labels, all_preds, average=None)
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        print(f1_each)
        return total_loss / len(val_dataset), f1


# 定义测试函数
def test_model(model, test_dataset, device, y_key=0, sequence_level=False
               , traditional_features=False, counter_features=False
               , return_event_importance=False
               ):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    trans_out_list = []
    with torch.no_grad():
        batch_episodes = test_dataset.intervals
        event_importance_results = []  # 每个episode一个dict：{'event_idx':..., 'event_importance':...}
        for episode in batch_episodes:
            episode_labels = torch.tensor([frame['global'].y[y_key] for frame in episode]).to(device)
            if torch.tensor(4) in episode_labels or torch.tensor(5) in episode_labels:
                episode_labels = (episode_labels == 5).long()

            if sequence_level:
                labels, _ = torch.mode(episode_labels)
                if return_event_importance:
                    logits, preds, transformer_out, _, _, attn, event_mask = model.predict_episode(
                        episode, device, sequence_level=True,
                        return_attn=True,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )
                    # attn: [1,H,T,T], event_mask: [T]
                    A = attn[0]  # [H,T,T]
                    event_idx = (event_mask == 1).nonzero(as_tuple=False).squeeze(-1)
                    if event_idx.numel() == 0:
                        event_importance = torch.zeros((0,), device=A.device)
                    else:
                        event_importance = A[:, :, event_idx].mean(dim=1).mean(dim=0)  # [K]

                    event_importance_results.append({
                        'event_idx': event_idx.detach().cpu(),
                        f'event_importance': event_importance.detach().cpu()
                    })
                else:
                    logits, preds, transformer_out, _, _ = model.predict_episode(
                        episode, device, sequence_level=True,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )

                labels = labels.cpu().numpy()
                preds = preds.cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)

            else:
                if return_event_importance:
                    logits, preds, transformer_out, _, _, attn, event_mask = model.predict_episode(
                        episode, device,
                        return_attn=True,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )
                    A = attn[0]  # [H,T,T]
                    event_idx = (event_mask == 1).nonzero(as_tuple=False).squeeze(-1)
                    if event_idx.numel() == 0:
                        event_importance = torch.zeros((0,), device=A.device)
                    else:
                        T = attn.size(-1)
                        event_importance = A[:, :, event_idx].mean(dim=1).mean(dim=0)

                    event_importance_results.append({
                        'event_idx': event_idx.detach().cpu(),
                        f'event_importance': event_importance.detach().cpu(),
                        "T": int(attn.size(-1))
                    })
                else:
                    logits, preds, transformer_out, _, _ = model.predict_episode(
                        episode, device,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )

                labels = episode_labels.view(-1).cpu().numpy()
                preds = preds.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

            trans_out_list.append(transformer_out.cpu())
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_each = f1_score(all_labels, all_preds, average=None)

    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test F1-score: {f1:.4f}")
    print(f1_each)
    print("Confusion Matrix:")
    print(cm)

    if return_event_importance:
        global_offset = 0

        all_global_event_id = []
        all_event_importance = []
        all_episode_id = []
        all_local_event_idx = []

        for epi_id, info in enumerate(event_importance_results):
            event_idx = info["event_idx"]  # shape [K]
            event_imp = info["event_importance"]  # shape [K]
            T = info["T"]

            global_event_id = event_idx + global_offset

            all_global_event_id.append(global_event_id)
            all_event_importance.append(event_imp)
            all_episode_id.append(torch.full_like(event_idx, epi_id))
            all_local_event_idx.append(event_idx)

            global_offset += T

        all_global_event_id = torch.cat(all_global_event_id, dim=0)
        all_event_importance = torch.cat(all_event_importance, dim=0)
        all_episode_id = torch.cat(all_episode_id, dim=0)
        all_local_event_idx = torch.cat(all_local_event_idx, dim=0)
        if y_key == 0:
            category = 'intention'
        else:
            category = 'phase'
        df = pd.DataFrame({
            "global_event_id": all_global_event_id.cpu().numpy(),
            # "episode_id": all_episode_id.cpu().numpy(),
            # "local_frame_idx": all_local_event_idx.cpu().numpy(),
            f"event_importance_{category}": all_event_importance.cpu().numpy(),
        })
        # df.to_csv("event_importance_all.csv", index=False)

        return all_preds, all_labels, f1, f1_each, df

    return all_preds, all_labels, f1, f1_each
        # , torch.cat(trans_out_list, dim=0)


def _event_importance_df(event_importance_results, y_key):
    global_offset = 0

    all_global_event_id = []
    all_event_importance = []

    for info in event_importance_results:
        event_idx = info["event_idx"]
        event_imp = info["event_importance"]
        T = info["T"]

        if event_idx.numel() > 0:
            all_global_event_id.append(event_idx + global_offset)
            all_event_importance.append(event_imp)

        global_offset += T

    category = 'intention' if y_key == 0 else 'phase'
    if not all_global_event_id:
        return pd.DataFrame({
            "global_event_id": [],
            f"event_importance_{category}": [],
        })

    all_global_event_id = torch.cat(all_global_event_id, dim=0)
    all_event_importance = torch.cat(all_event_importance, dim=0)

    return pd.DataFrame({
        "global_event_id": all_global_event_id.cpu().numpy(),
        f"event_importance_{category}": all_event_importance.cpu().numpy(),
    })


def predict_model(model, test_dataset, device, y_key=0, sequence_level=False
                  , traditional_features=False, counter_features=False
                  , return_event_importance=False
                  ):
    model.to(device)
    model.eval()
    all_preds = []

    with torch.no_grad():
        batch_episodes = test_dataset.intervals
        event_importance_results = []
        for episode in batch_episodes:
            if sequence_level:
                if return_event_importance:
                    logits, preds, transformer_out, _, _, attn, event_mask = model.predict_episode(
                        episode, device, sequence_level=True,
                        return_attn=True,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )
                    A = attn[0]
                    event_idx = (event_mask == 1).nonzero(as_tuple=False).squeeze(-1)
                    if event_idx.numel() == 0:
                        event_importance = torch.zeros((0,), device=A.device)
                    else:
                        event_importance = A[:, :, event_idx].mean(dim=1).mean(dim=0)
                    event_importance_results.append({
                        "event_idx": event_idx.detach().cpu(),
                        "event_importance": event_importance.detach().cpu(),
                        "T": int(attn.size(-1)),
                    })
                else:
                    logits, preds, transformer_out, _, _ = model.predict_episode(
                        episode, device, sequence_level=True,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )

                all_preds.append(preds.cpu().numpy())

            else:
                if return_event_importance:
                    logits, preds, transformer_out, _, _, attn, event_mask = model.predict_episode(
                        episode, device,
                        return_attn=True,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )
                    A = attn[0]
                    event_idx = (event_mask == 1).nonzero(as_tuple=False).squeeze(-1)
                    if event_idx.numel() == 0:
                        event_importance = torch.zeros((0,), device=A.device)
                    else:
                        event_importance = A[:, :, event_idx].mean(dim=1).mean(dim=0)
                    event_importance_results.append({
                        "event_idx": event_idx.detach().cpu(),
                        "event_importance": event_importance.detach().cpu(),
                        "T": int(attn.size(-1)),
                    })
                else:
                    logits, preds, transformer_out, _, _ = model.predict_episode(
                        episode, device,
                        counter_features=counter_features,
                        traditional_features=traditional_features
                    )

                all_preds.extend(preds.cpu().numpy())

    if return_event_importance:
        return all_preds, _event_importance_df(event_importance_results, y_key)

    return all_preds


def predict_node(model, test_dataset, device, y_key=0, sequence_level=False):
    model.to(device)
    model.eval()
    all_label_probs = []
    all_labels = []
    batch_episodes = test_dataset.intervals
    total_loss = 0
    for episode in tqdm(batch_episodes):
        episode_labels = torch.tensor([frame['global'].y[y_key] for frame in episode]).to(device)
        if torch.tensor(4) in episode_labels or torch.tensor(5) in episode_labels:
            episode_labels = (episode_labels == 5).long()
        with torch.no_grad():
            if sequence_level:
                logits, preds, probs, _ = model.predict_episode(episode, device, node_level=True, sequence_level=True)
                sequence_target, _ = torch.mode(episode_labels)
                sequence_target = sequence_target.item()
            else:
                logits, preds, probs, _ = model.predict_episode(episode, device, node_level=True)
        preds = preds.cpu().numpy()
        probs = probs.cpu().numpy()
        if sequence_level:
            label_prob = probs[:, sequence_target]
            labels_broadcast = [sequence_target]*22
        else:
            # print(probs)
            labels = episode_labels.view(-1).cpu().numpy()
            # print(labels.shape)
            label_prob = probs[np.arange(labels.shape[0])[:, None], np.arange(probs.shape[1])[None, :], labels[:, None]]
            labels_broadcast = np.broadcast_to(labels[:, None], (labels.shape[0], probs.shape[1]))
        # print(label_prob)
        all_label_probs.append(label_prob)
        all_labels.append(labels_broadcast)

    return all_label_probs, all_labels


def predict_node_attention(model, test_dataset, device, y_key=0, sequence_level=False):
    model.to(device)
    model.eval()
    all_attentions = []
    all_temporal_atts = []
    all_labels = []
    batch_episodes = test_dataset.intervals
    total_loss = 0
    for episode in tqdm(batch_episodes):
        episode_labels = torch.tensor([frame['global'].y[y_key] for frame in episode]).to(device)
        episode_frame_indices = torch.tensor([frame['Frame_index'].id for frame in episode]).to(device)
        # print(episode_labels.max())
        if torch.tensor(4) in episode_labels or torch.tensor(5) in episode_labels:
            episode_labels = (episode_labels == 5).long()
        with torch.no_grad():
            if sequence_level:
                logits, preds, probs, attention, temporal_att = model.predict_episode(episode, device, node_level=False, sequence_level=True)
                sequence_target, _ = torch.mode(episode_labels)
                sequence_target = sequence_target.item()
            else:
                logits, preds, probs, attention, temporal_att = model.predict_episode(episode, device, node_level=False
                                                                                   , counter_features=True
                                                                                      , traditional_features=True)
                # attention[attention == 0] = float('nan')
                # print(len(attention))
        if sequence_level:
            # label_prob = probs[:, sequence_target]
            labels_broadcast = [sequence_target]*22
        else:
            # print(probs)
            labels = episode_labels.view(-1).cpu().numpy()
            # print(labels.shape)
            # label_prob = probs[np.arange(labels.shape[0])[:, None], np.arange(probs.shape[1])[None, :], labels[:, None]]
            labels_broadcast = np.broadcast_to(labels[:, None], (labels.shape[0], probs.shape[1]))

        # print(label_prob)
        all_attentions.append(attention)
        all_labels.append(labels_broadcast)
        all_temporal_atts.append(temporal_att)

    return all_attentions, all_temporal_atts, all_labels
