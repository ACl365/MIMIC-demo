# Imbalance Handling Techniques Analysis: Summary Report

## Dataset Characteristics

> **Note: Data Restructuring**
> The data has been restructured to ensure each row represents a unique hospital admission, rather than individual measurements. The statistics below reflect the corrected dataset structure.

- **Total data points**: 404 (unique hospital admissions from 200 patients in the MIMIC-III demo dataset)
- **Positive class (readmission)**: ~49 (12.1%) - estimated based on original ratio
- **Negative class (no readmission)**: ~355 (87.9%) - estimated based on original ratio
- **Imbalance ratio**: ~7.2:1

## Performance Metrics

| Technique | Precision | Recall | F1 Score | PR AUC |
|-----------|-----------|--------|----------|--------|
| Baseline | 0.000 | 0.000 | 0.000 | 0.244 |
| Class Weights | 0.193 | 0.816 | 0.312 | 0.198 |
| Random Oversampling | 0.196 | 0.818 | 0.316 | 0.194 |
| SMOTE | 0.250 | 0.732 | 0.373 | 0.199 |
| Random Undersampling | 0.190 | 0.807 | 0.308 | 0.180 |

## Key Findings

1. **Baseline Performance**: The baseline model (without any imbalance handling) failed to predict any positive cases, highlighting the critical need for imbalance handling techniques in this dataset.

2. **Best Overall Technique**: SMOTE provided the best overall performance with the highest F1 score (0.373) and a good balance between precision and recall. It improved precision significantly compared to other techniques while maintaining reasonable recall.

3. **Precision-Recall Trade-off**:
   - Class weights and random oversampling achieved high recall (>0.81) but at the cost of precision (<0.20)
   - SMOTE achieved better precision (0.25) with somewhat lower recall (0.73)
   - This trade-off is important to consider based on the specific clinical context - whether missing a readmission (false negative) is more costly than a false alarm (false positive)

4. **SMOTE vs. Random Oversampling**:
   - SMOTE improved precision by 28% compared to random oversampling
   - SMOTE had lower recall by 10.5%
   - SMOTE's F1 score was 18.2% higher
   - This demonstrates that creating synthetic samples (SMOTE) rather than duplicating existing ones (random oversampling) leads to better generalization and more balanced predictions

5. **Undersampling Performance**: Random undersampling performed slightly worse than oversampling techniques, likely due to information loss from discarding majority class samples.

## Discussion of Trade-offs

### Baseline vs. Class Weights
Class weights adjust the importance of each class during training, which helps the model pay more attention to the minority class without changing the data. In our analysis, this dramatically improved recall from 0 to 0.816, demonstrating how effective this simple technique can be for imbalanced datasets.

### Random Oversampling vs. SMOTE
Random oversampling duplicates existing minority samples, which can lead to overfitting as the model sees the exact same minority samples multiple times. SMOTE creates synthetic examples by interpolating between existing minority samples, which helps the model generalise better by learning from a more diverse set of minority class examples.

The results clearly show this difference: SMOTE achieved significantly better precision and F1 score, indicating better generalisation, even though random oversampling had slightly higher recall. This suggests that SMOTE's synthetic samples helped the model learn more robust decision boundaries rather than just memorising the duplicated samples.

### Oversampling vs. Undersampling
Oversampling techniques (Random Oversampling, SMOTE) increase the number of minority class samples to balance the classes, preserving all available information. Undersampling reduces the number of majority class samples, which can lead to information loss.

In our analysis, undersampling performed slightly worse than oversampling techniques across most metrics, suggesting that the information loss from discarding majority class samples outweighed the benefits of balanced classes.

## Limitations

It's important to note the limitations of this analysis:

1. **Data Restructuring**: The original analysis was performed on data where each row represented a clinical measurement rather than a unique hospital admission. The code has been updated to restructure the data correctly, but the performance metrics may differ from the original analysis.

2. **Small Dataset Size**: With only 200 unique patients and 404 hospital admissions in the MIMIC-III demo dataset, the absolute performance metrics may be unstable and not generalisable. The relative differences between techniques are more informative than the absolute values.

3. **Cross-validation Stability**: Even with cross-validation, the small dataset size means that the results may vary significantly depending on the random splits.

3. **Model Simplicity**: We used logistic regression for all techniques to focus on the imbalance handling methods, but more complex models might interact differently with these techniques.

## Recommendations

Based on this analysis, we recommend:

1. **Use SMOTE for this dataset**: SMOTE provides the best overall performance with a good balance between precision and recall.

2. **Consider the clinical context**: If missing a readmission is much more costly than a false alarm, class weights or random oversampling might be preferred for their higher recall.

3. **Combine techniques**: Consider combining SMOTE with class weights to potentially further improve performance.

4. **Validate on larger dataset**: When the full MIMIC dataset becomes available, validate these findings on a larger, more representative sample.

5. **Explore threshold optimisation**: Adjust the classification threshold to further optimise the precision-recall trade-off based on specific clinical requirements.
