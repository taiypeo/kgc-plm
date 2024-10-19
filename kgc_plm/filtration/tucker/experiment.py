import logging
import time
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.optim.lr_scheduler import ExponentialLR

from ...graphs import BaseGraph
from .model import TuckER


logger = logging.getLogger(__name__)


class TuckERExperiment:
    def __init__(
        self,
        learning_rate: float = 0.0005,
        ent_vec_dim: int = 200,
        rel_vec_dim: int = 200,
        num_iterations: int = 500,
        batch_size: int = 128,
        decay_rate: float = 0.0,
        cuda: bool = False,
        input_dropout: float = 0.3,
        hidden_dropout1: float = 0.4,
        hidden_dropout2: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {
            "input_dropout": input_dropout,
            "hidden_dropout1": hidden_dropout1,
            "hidden_dropout2": hidden_dropout2,
        }

        if self.cuda:
            logging.info("Using CUDA for TuckER training and evaluation")

    def get_data_idxs(self, data: Dataset) -> list[tuple[int, int, int]]:
        transformed_data = data.map(
            lambda x: {
                "result": (
                    self.entity_idxs[x["head"]],
                    self.relation_idxs[x["relation"]],
                    self.entity_idxs[x["tail"]],
                )
            }
        )
        return [tuple(triple) for triple in transformed_data["result"]]

    def get_er_vocab(
        self, data: list[tuple[int, int, int]]
    ) -> dict[tuple[int, int], int]:
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return dict(er_vocab)

    def get_batch(
        self,
        er_vocab: dict[tuple[int, int], int],
        er_vocab_pairs: list[tuple[int, int]],
        idx: int,
        n_entity_ids: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = er_vocab_pairs[idx : idx + self.batch_size]
        targets = np.zeros((len(batch), n_entity_ids))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.0
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model: TuckER, graph: BaseGraph, data: Dataset) -> None:
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(
            self.get_data_idxs(
                concatenate_datasets([
                    graph.triplets[split_name] for split_name in graph.triplets
                ])
            )
        )

        logger.info("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(
                er_vocab, test_data_idxs, i, len(graph.entity_ids)
            )
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info("Hits @10: {0}".format(np.mean(hits[9])))
        logger.info("Hits @3: {0}".format(np.mean(hits[2])))
        logger.info("Hits @1: {0}".format(np.mean(hits[0])))
        logger.info("Mean rank: {0}".format(np.mean(ranks)))
        logger.info("Mean reciprocal rank: {0}".format(np.mean(1.0 / np.array(ranks))))

    def train_and_eval(self, graph: BaseGraph) -> TuckER:
        logger.info("Training the TuckER model...")
        self.entity_idxs = {
            entity_id: i for i, entity_id in enumerate(graph.entity_ids)
        }
        self.relation_idxs = {relation: i for i, relation in enumerate(graph.relations)}

        train_data_idxs = self.get_data_idxs(graph.triplets["train"])
        logger.info("Number of training data points: %d" % len(train_data_idxs))
        logger.info("Number of training epochs: %d" % self.num_iterations)

        model = TuckER(graph, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        logger.info("Starting training...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(
                    er_vocab, er_vocab_pairs, j, len(graph.entity_ids)
                )
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (
                        1.0 / targets.size(1)
                    )
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            logger.info(it)
            logger.info(time.time() - start_train)
            logger.info(np.mean(losses))
            model.eval()
            with torch.no_grad():
                logger.info("Validation:")
                self.evaluate(model, graph, graph.triplets["validation"])
                if not it % 2:
                    logger.info("Test:")
                    start_test = time.time()
                    self.evaluate(model, graph, graph.triplets["test"])
                    logger.info(time.time() - start_test)

        return model
