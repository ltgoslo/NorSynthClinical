# NorSynthClinical

Annotation guidelines for family history extraction from synthetic corpus of clinical text. The paper titled **Iterative development of family history annotation guidelines using a
synthetic corpus of clinical text** is accepted for publication at [LOUHI](https://louhi2018.fbk.eu/home) workshop which is collocated with [EMNLP 2018](http://emnlp2018.org).

## Requirements
- Scikit-learn

## Code and data for experiments 
The results reported in the paper can be replicated by 

<!--Compute IAA between two annotators is computed as given below. We treat the data annotated by the clinician as gold standard. In this case, the clinician is Pål Brekke.
- `python3 interannotator_agreement.py taraka_annotate pal_annotate`
- `python3 interannotator_agreement.py lilja_annotate pal_annotate`-->

Train and test SVM 5-fold cross-validation for entity recognition
- `python3 svm_ner.py all_sentences.vert.parse.entity 5`

Train and test SVM 5-fold cross-validation for relation extraction. The file 
- `python3 uio2rel.py pal_annotate all_sentences.vert.parse 5`
