# IND-DSP-Improving-Night-Domain-generalization-segmentation-via-Diffusion-Semantic-Prior

We use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
Requirements for module required is inside the segmentation folder.
And also required to get the segmentation logits from the RobustNet model 

## Training with our modified training and Validation (Sampling) procedure
bash tools/dist_train.sh configs/cityscapes/ddp_convnext_t_4x4_512x1024_160k_cityscapes.py 4 

## Infer on Cityscapes Dataset
bash tools/dist_test.sh configs/cityscapes/ddp_convnext_t_4x4_512x1024_160k_cityscapes.py /raid/ai24resch01002/saved_models/ddp/ddp_convnext_t_4x4_512x1024_160k_cityscapes_best_mod_training/best_mIoU_iter_8000.pth 4  --eval mIoU 

## Infer on DarkZurich Dataset
CUDA_VISIBLE_DEVICES=0,1,2,5 bash tools/dist_test.sh configs/darkzurich/ddp_convnext_t_4x4_1080x1920_test_darkzurich_mod_ours.py /raid/ai24resch01002/saved_models/ddp/ddp_convnext_t_4x4_512x1024_160k_cityscapes_best_mod_training/best_mIoU_iter_8000.pth 4  --eval mIoU






