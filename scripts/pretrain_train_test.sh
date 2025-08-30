# qwen3
# pretrain 
#python /root/test/train.py /root/test/config/pretrain_QI/pretrain_QI.json
# fine tuning
python /root/test/train.py /root/test/config/train_pretrain_QI/train_pretrain_QI.json
# test
python /root/test/train.py /root/test/config/predict_QI/predict_QI_XLMR.json


# qwen3-reranker
# pretrain
#python /root/test/train.py /root/test/config/pretrain_QI/pretrain_QI_reranker.json
# fine tuning
python /root/test/train.py /root/test/config/train_pretrain_QI/train_pretrain_QI_reranker.json
# test
python /root/test/train.py /root/test/config/predict_QI/predict_QI_XLMR_reranker.json


# qwen3-embedding
# pretrain
#python /root/test/train.py /root/test/config/pretrain_QI/pretrain_QI_emb.json
# fine tuning
python /root/test/train.py /root/test/config/train_pretrain_QI/train_pretrain_QI_emb.json
# test
python /root/test/train.py /root/test/config/predict_QI/predict_QI_XLMR_emb.json