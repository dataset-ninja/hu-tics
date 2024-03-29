The authors created **HuTics: Human Deictic Gestures Dataset**, consisting of 2040 images collected from 170 people that include various deictic gestures and objects with segmentation mask annotations. The technical evaluation shows that their object highlights can achieve the accuracy of 0.718 (mean Intersection over Union; 𝑚𝐼𝑜𝑈) and can run at 28.3 fps. It covers four kinds of deictic gestures to objects: ***exhibit***, ***pointing***, ***present*** and ***touch***.

## Motivation

Interactive Machine Teaching (IMT) endeavors to enhance users' teaching involvement in Machine Learning (ML) model creation. Tailored for non-ML experts, IMT systems enable users to contribute training data through demonstrations. Among these, Vision-based IMT (V-IMT) systems utilize cameras to capture users' demonstrations. For instance, Teachable Machine allows users to construct a computer vision classification model by presenting various perspectives of each object (class) to a camera.

Despite the minimal effort required to provide training samples, prior research has indicated that ML models trained via V-IMT systems might identify objects based on irrelevant visual features. For instance, during a demonstration of a book, the model may focus on background visual elements. Failure to rectify this issue could lead to diminished model performance in real-world applications. Hence, users should be empowered to specify the image portions that the model must prioritize for accurate classification. One strategy to address this is through object annotations, which can be incorporated into the model's training process.

Advancements in annotation tools streamline user tasks, simplifying interactions to clicks or sketches. Despite their reduced workload, existing annotation tools are not optimized for V-IMT systems, necessitating users to conduct annotations post hoc. This undermines the overall user experience with V-IMT systems. Consequently, exploring annotation methods more seamlessly integrated into V-IMT systems is crucial.

The backend model responsible for identifying object highlights requires training data illustrating how individuals gesture towards objects in front of a camera. Among the available datasets focusing on human-object interactions, [TEgO](https://dl.acm.org/doi/10.1145/3290605.3300566) stands out as the most suitable for the authors' task. TEgO comprises 5758 labeled egocentric images capturing hand-object interactions. Each image includes a hand segmentation mask and precise point-level annotations indicating the object's location. However, these annotations alone do not fulfill the requirement for object segmentation. To address this, an attempt was made to infer the segmentation mask of the object using a click-based interactive segmentation approach. Subsequently, all generated results underwent manual inspection to eliminate data samples where the inferred segmentation masks proved to be significantly inaccurate. This process led to the creation of a custom dataset, named TEgO-Syn, comprising 5232 instances with automatically synthesized object segmentation masks.

The authors observations showed that the model was not robust enough. They then summarized three main reasons why TEgO still cannot fit their target task:

* **A limited set of gestures.** All data in TEgO were collected from two participants, which is insufficient to cover how different people interact with the object using gestures.
* **A limited set of objects.** TEgO-Syn includes 5232 images of 19 objects. Training on a small set of objects repeatedly enables the model to over-fit the features of these specific objects, which is harmful to our target task, i.e., object-agnostic segmentation.
* **Egocentric images.** The images in the TEgO dataset are taken from the egocentric view. Our system uses a front-facing camera, which is a common configuration in V-IMT.



## Dataset description

The authors created the HuTics dataset consisting of 2040 images collected from 170 people that include various deictic gestures and objects with segmentation
mask annotations. They leveraged crowd-workers from Amazon Mechanical Turk to enrich the dataset's diversity, following approval from their university for data collection. Each task entailed uploading 12 images depicting how deictic gestures are utilized to reference objects clearly. To ensure a varied image set, the authors categorized deictic gestures into four types: ***pointing***, ***present***, ***touch***, and ***exhibit***. Workers were instructed to capture three distinct photos for each gesture category, supplemented with example images for clarity. In total, 2040 qualifying images were amassed from 170 crowd-workers, averaging 34 years in age. Unacceptable submissions included excessively blurry images or those devoid of any discernible gestures.

On average, crowd-workers dedicated 15 minutes per task, receiving $2 for their participation. Additionally, the authors engaged five individuals through a local crowdsourcing platform to annotate object segmentation masks on the gathered images. Each annotation worker labeled approximately 408 images and received an average compensation of $78 in local currency. [AnnoFab](https://annofab.com/), an online polygon-based tool, facilitated the annotation process for delineating segmentation masks.

<img src="https://github.com/dataset-ninja/hu-tics/assets/120389559/5d59496c-626c-4b86-9865-136911ca8670" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">Example images in HuTics dataset. HuTics covers four kinds of deictic gestures to objects: exhibiting (top-left), pointing (top-right), presenting (bottom-left) and touching (bottom-right). The hands and objects of interest are highlighted in blue and green, respectively.</span>

