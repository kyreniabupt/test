
# emb
# stage1
python /root/test/train.py /root/test/config/gradually/QC/stage1_QC_emb.json
# stage2
python /root/test/train.py /root/test/config/gradually/QC/stage2_QC_emb.json
# predict
python /root/test/train.py /root/test/config/gradually/QC/predict_QC_emb.json



# reranker
# stage1
#python /root/test/train.py /root/test/config/gradually/QC/stage1_QC_reranker.json
# stage2
#python /root/test/train.py /root/test/config/gradually/QC/stage2_QC_reranker.json
# predict
#python /root/test/train.py /root/test/config/gradually/QC/predict_QC_reranker.json