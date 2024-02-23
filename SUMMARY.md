**HuTics Dataset** is a dataset for instance segmentation, semantic segmentation, object detection, and identification tasks. It is used in the robotics industry. 

The dataset consists of 2040 images with 6428 labeled objects belonging to 2 different classes including *object of interest* and *hand*.

Images in the HuTics dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 5 (0% of the total) unlabeled images (i.e. without annotations). There are 2 splits in the dataset: *train* (1632 images) and *test* (408 images). Alternatively, the dataset could be split into 4 taxonomy: ***present*** (1728 objects), ***pointing*** (1645 objects), ***touch*** (1563 objects), and ***exhibit*** (1492 objects). Additionally, every image marked with its ***sequence*** tag. The dataset was released in 2022 by the University of Tokyo, Japan.

<img src="https://github.com/dataset-ninja/hu-tics/raw/main/visualizations/poster.png">
