# WCS-Team-3

### How to build the model
1. [Background](https://towardsdatascience.com/neural-style-transfer-tutorial-part-1-f5cd3315fa7f)
2. [Implementation](https://towardsdatascience.com/neural-style-transfer-series-part-2-91baad306b24)

### Files
download\_vgg19.py: Downloads pretrained VGG-19 and stores in local filepath. I believe doing so should speed up time to load pretrained model but I could be wrong.<br />
modified\_vgg.py: Declare modified VGG (average instead of max pool), transfer pretrained weights, save model.
