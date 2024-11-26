# Word Sense Disambiguation
## Problem Setup
This experiment investigates Word Sense Disambiguation (WSD), an NLP task that determines a word's meaning based on context. Several methods were evaluated: the Lesk algorithm, a graph-based approach, and embedding-based techniques using PMI and the pre-trained model GlossBERT. The objective is to achieve a good accuracy in predicting word meanings while addressing two key questions:

1. What improvements can enhance each method's performance?
2. What limitations hinder these methods from achieving higher accuracy?

## Experimental Setup
This study uses the SemEval 2013 Shared Task #12 dataset (Navigli and Jurgens, 2013), featuring texts from multilingual-all-words.en.xml and annotated senses in wordnet.en.key, with NLTK's WordNet interface as the lexical resource. Each WSD method was implemented separately to enable parallel development. The dataset was preprocessed by loading instances, matching them to their correct senses, filtering incomplete data, and splitting it into 194 development and 1450 test instances. Accuracy was calculated for each method using evaluation functions.

## Model Descriptions
### Baseline Method
The baseline, implemented in Method1.py, predicts the most frequent sense of a word by selecting the first synset in WordNet for a given lemma using the wordnet.synsets function. This straightforward approach requires no training or hyperparameter tuning, making it a quick benchmark for WSD tasks. Below is a sample output comparing the predicted and gold labels from the baseline evaluation.<br>
* d002.s001.t001: Lemma=victory, Predicted=victory%1:11:00::, Gold=victory%1:11:00::
* d002.s002.t006: Lemma=loss, Predicted=loss%1:21:01::, Gold=loss%1:04:00::

### Lesk Algorithm
The Lesk algorithm from nltk.wsd, found in Method1.py, was enhanced with POS tagging, defaulting missing tags to nouns for improved performance. A configurable filter_text variable controlled preprocessing: enabling it removed stop words and punctuation, while disabling it preserved the raw context. Experiments revealed that lowercasing slightly reduced development accuracy (~0.18%) and was therefore excluded. The unfiltered approach (filter_text=False) outperformed the filtered version on the development set (37.11% vs. 35.57%), leading to its selection for test evaluation. Below is a sample output comparing predicted and gold labels using the Lesk algorithm.<br>
* d002.s001.t001: Lemma: victory, Predicted: victory%1:11:00::, Gold: victory%1:11:00::
* d002.s002.t006: Lemma: loss, Predicted: loss%1:21:00::, Gold: loss%1:04:00::

### Graph, PMI, and Hybrid Solutions
The second solution for word sense disambiguation (WSD), implemented in Method2.py, builds upon the ideas outlined in [1]. This graph-based, unsupervised method exploits semantic relationships in WordNet, such as hypernyms, hyponyms, and meronyms, to form a graph of interconnected synsets. Unlike [1], which prioritizes semantic relatedness measures, this approach applies PageRank to compute centrality scores, predicting the sense with the highest rank as the most appropriate meaning.
To improve the graph-based solution, a PMI-based method was introduced to generate embeddings for synsets. By calculating cosine similarity between context vectors and synset embeddings, the synset with the highest similarity score was selected as the predicted sense. This PMI method was first evaluated as a standalone approach before being integrated into a hybrid method alongside the graph-based approach.

The hybrid approach combined graph centrality with PMI embeddings, ranking synsets through a blend of centrality and similarity scores, achieving a ~5% improvement over the graph-based method. Interestingly, the standalone PMI-based method outperformed both the hybrid and graph-based approaches, achieving ~3% and ~7% higher development accuracy, respectively. This advantage is likely due to its exclusive focus on semantic similarity, avoiding the potential noise introduced by combining graph-based centrality. On the development set, the graph-based, PMI-based, and hybrid methods achieved accuracies of 43.30%, 50.52%, and 47.94%, respectively. With the PMI-based method demonstrating the best performance, it was further evaluated on the test set. The sample outputs below show the predicted and gold labels for each method. <br>
* Graph-based d002.s001.t001: Instance=victory, Prediction=victory%1:11:00::, Gold=victory%1:11:00::
* Graph-based d002.s002.t002: Instance=loss, Prediction=inflict%2:32:00::, Gold=loss%1:04:00::
* PMI-based d002.s001.t001: Instance=victory, Prediction=victory%1:11:00::, Gold=victory%1:11:00::
* PMI-based d002.s002.t006: Instance=loss, Prediction=loss%1:21:01::, Gold=loss%1:04:00::
* Hybrid d002.s001.t001: Instance=victory, Prediction=victory%1:11:00::, Gold=victory%1:11:00::
* Hybrid d002.s002.t006: Instance=loss, Prediction=loss%1:21:01::, Gold=loss%1:04:00::

### kanishka/GlossBERT Pre-Trained Online LLM from Hugging Face
Building on the successful use of embeddings in the previous section, GlossBERT, a pre-trained transformer model from [2], was further fine-tuned for the WSD task using PyTorch. Regularization techniques—including the AdamW optimizer with weight decay, gradient clipping, and a learning rate scheduler—were applied to mitigate overfitting. Training was limited to 4 epochs, yielding optimal results. GlossBERT achieved the best performance, with a development accuracy of 76.41% and a test accuracy of 57.73%, surpassing all prior non-baseline solutions. Sample outputs are provided below.<br>
* d001.s001.t002: Lemma: group, Predicted: group%1:03:00::, Gold: group%1:03:00::
* d001.s002.t001: Lemma: climate, Predicted: climate%1:26:01::, Gold: climate%1:26:00::

## Analysis of Results
| Model                | Development Accuracy | Test Accuracy |
|----------------------|----------------------|---------------|
| Baseline             | N/A                  | 62.34%        |
| Lesk                 | 37.11%               | 38.41%        |
| PMI-embeddings       | 50.52%               | 49.38%        |
| GlossBERT            | 76.41%               | 57.73%        |

The final results above highlight the strengths and limitations of the WSD approaches. These results show the baseline model's strength in aligning with common word senses, reflecting typical usage patterns. However, its reliance on the most frequent sense limits its ability to handle rare, context-specific meanings, reducing its effectiveness in nuanced scenarios where less frequent interpretations are required.

The Lesk algorithm achieved 38.41% accuracy with minimal preprocessing and fast execution. Its poor performance likely stems from its simplicity. Enhancements such as more robust preprocessing, incorporating contextualized embeddings such as Word2Vec, and leveraging similarity metrics instead of word overlap could improve its effectiveness.

The PMI-embeddings method outperformed Lesk with 49.38% accuracy but faced challenges in ambiguous contexts due to the inherent limitations of PMI. Since PMI relies heavily on co-occurrence statistics, it struggles to capture nuanced relationships in sparse or ambiguous data. Additionally, the method treats context words independently, ignoring word order and syntactic structure, which are crucial for disambiguating complex meanings. Combining PMI with dimensionality reduction techniques like SVD could help address some of these issues by extracting more meaningful patterns from the embeddings, potentially improving performance.

GlossBERT, as a pre-trained WSD model, achieved a generalization accuracy of 57.73% after additional fine-tuning on the SemEval dataset. However, it falls short of state-of-the-art performance, suggesting room for improvement through more extensive fine-tuning, enhanced preprocessing strategies, or the integration of richer lexical resources to better handle rare or complex senses.


## Further Improvements
The study faced limitations, including a small dataset (1644 instances), which restricted models like GlossBERT; expanding the dataset could enhance performance. Relying solely on accuracy is also insufficient; incorporating metrics like precision, recall, and F1-score would allow a more nuanced evaluation of frequent and rare senses. Finally, the Lesk, graph-based, and PMI-embedding methods could all benefit from richer semantic resources, refined embedding schemes, and better feature extraction methodologies, while GlossBERT might improve further with some dropout and extended fine-tuning.

## References
[1] M. Arab, M. Z. Jahromi and S. M. Fakhrahmad, "A graph-based approach to word sense disambiguation. An unsupervised method based on semantic relatedness," 2016 24th Iranian Conference on Electrical Engineering (ICEE), Shiraz, Iran, 2016, pp. 250-255, doi: 10.1109/IranianCEE.2016.7585527.

[2] L. Huang, C. Sun, X. Qiu, and X. Huang, "GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, Nov. 2019, pp. 3507–3512. doi: 10.18653/v1/D19-1355. [Online]. Available: https://www.aclweb.org/anthology/D19-1355

[3] S. Khalid, "BERT Explained: A Complete Guide with Theory and Tutorial," Medium, Nov. 3, 2019. [Online]. Available: https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c