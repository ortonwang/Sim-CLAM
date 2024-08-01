code for "Prediction of PD-L1 and EBV expression in gastric cancer from H$\&$E-stained Whole Slide Imaging using deep learning"  
our code is origin from [CLAM]([https://github.com/ycwu1997/MC-Net](https://github.com/mahmoodlab/CLAM))  
To train model, run  
python main_TCGA_STAD.py  
the dataname in dataset_csv/TCGA_label.csv is for TCGA-STAD datset.  
The model_our_TCGA_STAD.pth is used for feature extracting.  
For our SimCLR self-supervised learning, you could refer https://github.com/vkola-lab/tmi2022
