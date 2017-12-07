# text_classification_hierarchical_cnn
Text classification using char-CNN + word-CNN

# Reference:
1. Convolutional Neural Networks for Sentence Classification
http://www.aclweb.org/anthology/D14-1181

2. Character-Aware Neural Language Models
https://arxiv.org/pdf/1508.06615.pdf


# File configuations:
1. Assign suitable name to variable 'dataset_identity' (dataset specific name) under class DataConfig. Create folder with name "dump_" + dataset_identity under 'data' folder. All dataset specific vocab, embedding_matrix will be automatically stored in this folder. It helps maintaining consistency while doing experiments on multiple datasets.
2. Like above, copy train, valid and test files under folder with name as 'dataset_identity' variable's value under 'data' folder and accordingly change file/folder paths into class DataConfig under 'utils/feature_extraction.py'.

# Hyperparameters/architecture related configurations:
1. Model and architecture related settings (number of layers, filters, enable fully connected layers, dropout, epochs, batch_size, lr etc..) can be adjusted via class 'ModelConfig' under 'utils/feature_extraction.py'
2. Current code uses SENNA 50d embeddings - https://ronan.collobert.com/senna/ however you can use your own embeddings. Just change filename via variable name 'embedding_file' into class 'DataConfig'

# Traininig/Testing => train.py

# For Training:
main(Flags.TRAIN, load_existing_dump=False)
    
'load_existing_dump': 
If set to False, will create vocabs, embedding_matrix etc. from input dataset and saves into 'dump_dir' as mentioned above.
If True, will load it from existing 'dump_dir' without creating vocabs again, thus little faster. It is useful while performing multiple training with same dataset: 
main(Flags.TRAIN, load_existing_dump=True)

# For Testing:
main(Flags.TEST, load_existing_dump=True)

# Error Analysis:
Incorrect test predictions will automatically get written under model_saver directory.
