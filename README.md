
Cardiovascular Disease Detection and Left Ventricle Segmentation
Cardiovascular disease stands as the leading cause of death globally, constituting nearly 40% of all fatalities in Canada. This project focuses on the visualization of the left ventricle, a critical aspect in assessing the ejection fraction metric. Ejection fraction, measuring the percentage of blood leaving the heart in each contraction, provides vital information to physicians regarding the severity of the condition and guides the formulation of an appropriate treatment plan.

Objective
The challenge lies in the manual assessment of ejection fraction by clinicians, a laborious task prone to errors due to inaccuracies in ventricle segmentation. To address this issue, we present a solution using a U-Net-based deep learning model. This model is trained on anonymized patient data to automatically generate accurate left ventricle segmentations from echocardiogram ultrasounds.

Implementation
Our approach involves implementing and training a U-Net-based deep learning model on a dataset of anonymized patient echocardiograms. The trained model provides automated left ventricle segmentations, reducing the reliance on manual assessment and potential human error.

Model Evaluation
To validate our model's performance, we quantitatively assess it against expert-segmented ground truths. After just two epochs of training, our model achieves a validation mean Intersection over Union (IoU) score of 0.84 and a mean Dice similarity coefficient of 0.91. These metrics demonstrate the model's accuracy in left ventricle segmentation.

Live Demonstration
For a visual representation of our system's performance, a live video demonstration is available here.

Conclusion
Through this research, we successfully implement a deep learning architecture capable of generating anatomically accurate left ventricle segmentations on ultrasound images. This automated approach holds the potential to enhance the efficiency and accuracy of cardiovascular disease diagnosis and treatment planning.

Note: This README provides an overview of our project. Please refer to the provided live video demonstration for a visual representation of our system's performance.
