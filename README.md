# scverse hackathon proteomics segmentation
30-31.03.26 scverse proteomics hackathon - segmentation project

Nikita Moshkov, Friedrich Preusser, Omar Elakad

Work with [CRC Schuerch dataset (2020)](https://linkinghub.elsevier.com/retrieve/pii/S0092-8674(20)30870-9)
[Single image to download.](https://drive.google.com/file/d/1h-DKw5c43v5DXcfyhH19IL8pLBH8AtpE/view?usp=drive_link)

Main tasks:
1. Segment TMA's with Mesmer.
2. Measure which cells are positive for mutually exclusive markers (use standard deviation as a threshold).
3. QC wrt to cell area and DRAQ5 intensity in the cell region.
_______
Uncertanity estimation: 
1. Segment rotate and flipped versions of images to evaluate uncertanity (test-time augmentation).
2. Measure agreement of segmentations.
_______
