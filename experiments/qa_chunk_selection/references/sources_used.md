# Sources used by experiments/qa_chunk_selection

These claims are adapted to in-document extractive QA on SciREX. They are not
literal reproductions of the original open-domain setups.

- source_id: S01
  title: RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering
  authors: Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Xin Zhao, Daxiang Dong, Hua Wu, Haifeng Wang
  year: 2020
  claim_used: denoised hard negatives for training
  used_in: [B]

- source_id: S02
  title: Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering
  authors: Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, Bing Xiang
  year: 2019
  claim_used: globally normalize answer scores across passages of the same question
  used_in: [C]

- source_id: S03
  title: Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering
  authors: Shuohang Wang, Mo Yu, Jing Jiang, Wei Zhang, Xiaoxiao Guo, Shiyu Chang, Zhiguo Wang, T. Klinger, G. Tesauro, Murray Campbell
  year: 2017
  claim_used: aggregate evidence from different passages
  used_in: [D5]

- source_id: S04
  title: Dense Hierarchical Retrieval for Open-Domain Question Answering
  authors: Kazuma Hashimoto, Yingbo Zhou, Semih Yavuz, Caiming Xiong, Philip S. Yu
  year: 2021
  claim_used: combine local passage semantics with coarser document/section context
  used_in: [D2]

- source_id: S05
  title: Mitigating False-Negative Contexts in Multi-document Question Answering with Retrieval Marginalization
  authors: Ansong Ni, Matt Gardner, Pradeep Dasigi
  year: 2021
  claim_used: multiple admissible evidence contexts; do not treat alternative positives as hard negatives
  used_in: [B, C]

- source_id: S06
  title: Long Context Question Answering via Supervised Contrastive Learning
  authors: Avi Caciularu, Ido Dagan, J. Goldberger, Arman Cohan
  year: 2021
  claim_used: discriminate supporting evidence sentences from negatives
  used_in: [D4]

- source_id: S07
  title: Answerability in Retrieval-Augmented Open-Domain Question Answering
  authors: Rustam Abdumalikov, Pasquale Minervini, Yova Kementchedjhieva
  year: 2024
  claim_used: irrelevant excerpts can have high semantic overlap; relevance is not answerability
  used_in: [A, B]
