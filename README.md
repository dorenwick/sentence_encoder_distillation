# sentence_encoder_distillation
This is a script for distilling the following model = SentenceTransformer("all-MiniLM-L6-v2") into smaller dimension models. 

we reduce the popular sentence encoder: "all-MiniLM-L6-v2" which is 384 dimensional bert based sentence encoder (see sbert library or huggingface),
to lower dimensional models: [256, 128, 64, 48, 32, 16, 8] Scores for each model are:  

256 Dim Model STS Score: 0.8261079485369712
128 Dim Model STS Score: 0.816830899095272
64 Dim Model STS Score: 0.7966557914485638
48 Dim Model STS Score: 0.7757417906216884
32 Dim Model STS Score: 0.745332350705585
16 Dim Model STS Score: 0.665596114418915
8 Dim Model STS Score: 0.596041263085443

This script takes a few minutes on an average computer.
