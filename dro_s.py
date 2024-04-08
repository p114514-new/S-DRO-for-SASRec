import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss

from collections import defaultdict


class DRO_S(SequentialRecommender):
    def __init__(self, config, dataset):
        super(DRO_S, self).__init__(config, dataset)

        self.dataset = dataset

        # load parameters info
        self.alpha = config["alpha"]
        self.dataset_name = config["dataset"]
        self.eta = config["eta"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.num_groups = config["num_groups"]
        self.user_group_weights = torch.tensor([1.0] * self.num_groups, dtype=torch.float32)
        self.thresholds = config["thresholds"]
        if len(self.thresholds) != self.num_groups - 1:
            raise ValueError("The number of thresholds does not match the number of groups.")
        self.user_groups = self.get_different_user_groups()

        self.group_loss = torch.zeros(self.num_groups, dtype=torch.float32)

        self.device = config['device']

        # parameters initialization
        self.apply(self._init_weights)

    def get_different_user_groups(self):
        if self.dataset_name == "retailrocket-view":
            user_ids = self.dataset.inter_feat['visitor_id'].cpu().numpy() - 1
        else:
            user_ids = self.dataset.inter_feat['user_id'].cpu().numpy() - 1
        item_ids = self.dataset.inter_feat['item_id'].cpu().numpy() - 1
        num_users = max(user_ids) + 1
        num_items = max(item_ids) + 1

        # get the popularity of each item
        item_popularity = defaultdict(int)
        for item_id in item_ids:
            item_popularity[item_id] += 1
        item_popularity = dict(sorted(item_popularity.items(), key=lambda x: x[1], reverse=True))
        popular_item_index = list(item_popularity.keys())[:int(len(item_popularity) * 0.2)]

        # create a 2-D matrix to store user-item interactions
        user_item_matrix = np.zeros((num_users, num_items), dtype=bool)
        for i, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            user_item_matrix[user_id, item_id] = True

        popular_item_matrix = user_item_matrix[:, popular_item_index]

        # total number of interactions for each user
        user_interactions = np.sum(user_item_matrix, axis=1)

        # number of interactions with popular items for each user
        user_popular_interactions = np.sum(popular_item_matrix, axis=1)

        popular_interaction_ratio = user_popular_interactions / user_interactions

        # classify users into different groups based on interaction ratios
        user_groups = np.zeros(num_users, dtype=int)
        for i, threshold in enumerate(self.thresholds):
            user_groups[popular_interaction_ratio >= threshold] = i + 1

        print(f"Number of users in each group: {np.bincount(user_groups)}")

        return torch.tensor(user_groups, dtype=int)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        self.user_group_weights = self.user_group_weights.to(self.device)
        self.group_loss = self.group_loss.to(self.device)
        self.user_groups = self.user_groups.to(self.device)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_seq = interaction[self.USER_ID] - 1
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # get user classes
        user_group = self.user_groups[user_seq]
        # get the sum of the loss of each group
        group_loss = torch.zeros(self.num_groups, dtype=torch.float32).to(self.device)
        for i in range(self.num_groups):
            group_loss[i] = torch.sum(loss[user_group == i])

        # recalculate the weight of each group
        group_loss = group_loss.detach()
        self.group_loss = torch.add((1 - self.alpha) * self.group_loss, self.alpha * group_loss)
        self.user_group_weights = torch.exp(self.eta * self.group_loss) * self.user_group_weights
        self.user_group_weights /= torch.sum(self.user_group_weights)
        return 0.01*torch.mean(loss * self.user_group_weights[user_group]) + torch.mean(loss)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
