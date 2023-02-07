# GA-DTCDR
This is the model in "A Graphical and Attentional Framework for Dual-Target Cross-Domain Recommendation" (IJCAI2020).
GA-DTCDR is an optimized model for DTCDR ("DTCDR: A Framework for Dual-Target Cross-Domain Recommendation" in CIKM2019).
DTCDR is the first work for dual-target cross-domain recommendation. Compared with DTCDR, we improved the embedding strategy (from DMF/NeuMF to Graph Embedding) and combination strategy (from fixed combination operators to element-wise attention). 

As for the doc2vec code and the raw data including text information, I have shared the desensitization raw data at https://www.researchgate.net/publication/350793434_Douban_dataset_ratings_item_details_user_profiles_and_reviews. If you want to learn how to use Doc2vec, you can visit https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec. 

I have uploaded some of the pre-trained doc2vec embeddings and node2vec embeddings (the file sizes of others are larger than 50M, I cannot summit them to GitHub). As for other pre-trained embddings, you can generate them by our provided codes.

# Raw Douban Dataset (reviews, item details, user profiles, tags, and ratings)
Due to the size limit (the file size of raw dataset is too large), so I upload the raw dataset at ResearchGate.

Url: https://www.researchgate.net/publication/350793434_Douban_dataset_ratings_item_details_user_profiles_and_reviews

# Citations
If you want to use our code or dataset, you should cite the following papers (at least one paper) in your submissions.

@inproceedings{zhugraphical,
  title={A Graphical and Attentional Framework for Dual-Target Cross-Domain Recommendation},
  author={Zhu, Feng and Wang, Yan and Chen, Chaochao and Liu, Guanfeng and Zheng, Xiaolin},
  booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020},
  pages={3001--3008},
  year={2020}
}

@inproceedings{zhu2019dtcdr,
  title={DTCDR: A framework for dual-target cross-domain recommendation},
  author={Zhu, Feng and Chen, Chaochao and Wang, Yan and Liu, Guanfeng and Zheng, Xiaolin},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={1533--1542},
  year={2019}
}

# Running
(1) Tensorflow Version: 1.8.1

(2) Requirements:

pip install gensim

pip install node2vec

(3) Running:

python GA-DTCDR.py
