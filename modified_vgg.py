import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, out_params = None):
        out = {}
        
        out['1_1'] = F.relu(self.conv1_1(x))
        out['1_2'] = F.relu(self.conv1_2(out['1_1']))
        pooled = self.pool(out['1_2'])
        
        out['2_1'] = F.relu(self.conv2_1(pooled))
        out['2_2'] = F.relu(self.conv2_2(out['2_1']))
        pooled = self.pool(out['2_2'])
        
        out['3_1'] = F.relu(self.conv3_1(pooled))
        out['3_2'] = F.relu(self.conv3_2(out['3_1']))
        out['3_3'] = F.relu(self.conv3_3(out['3_2']))
        out['3_4'] = F.relu(self.conv3_4(out['3_3']))
        pooled = self.pool(out['3_4'])
        
        out['4_1'] = F.relu(self.conv4_1(pooled))
        out['4_2'] = F.relu(self.conv4_2(out['4_1']))
        out['4_3'] = F.relu(self.conv4_3(out['4_2']))
        out['4_4'] = F.relu(self.conv4_4(out['4_3']))
        poooled = self.pool(out['4_4'])
                             
        out['5_1'] = F.relu(self.conv5_1(pooled))
        out['5_2'] = F.relu(self.conv5_2(out['5_1']))
        out['5_3'] = F.relu(self.conv5_3(out['5_2']))
        out['5_4'] = F.relu(self.conv5_4(out['5_3']))
        final = self.pool(out['5_4'])
        
        return [out[param] for param in out_params]

if __name__ == "__main__":
	pre_trained_vgg = models.vgg19(pretrained=True, progress=True)

	model = VGG()

	model.conv1_1.weight.data = pre_trained_vgg.features[0].weight
	model.conv1_2.weight.data = pre_trained_vgg.features[2].weight
	model.conv2_1.weight.data = pre_trained_vgg.features[5].weight
	model.conv2_2.weight.data = pre_trained_vgg.features[7].weight
	model.conv3_1.weight.data = pre_trained_vgg.features[10].weight
	model.conv3_2.weight.data = pre_trained_vgg.features[12].weight
	model.conv3_3.weight.data = pre_trained_vgg.features[14].weight
	model.conv3_4.weight.data = pre_trained_vgg.features[16].weight
	model.conv4_1.weight.data = pre_trained_vgg.features[19].weight
	model.conv4_2.weight.data = pre_trained_vgg.features[21].weight
	model.conv4_3.weight.data = pre_trained_vgg.features[23].weight
	model.conv4_4.weight.data = pre_trained_vgg.features[25].weight
	model.conv5_1.weight.data = pre_trained_vgg.features[28].weight
	model.conv5_2.weight.data = pre_trained_vgg.features[30].weight
	model.conv5_3.weight.data = pre_trained_vgg.features[32].weight
	model.conv5_4.weight.data = pre_trained_vgg.features[34].weight

	torch.save(model.state_dict(), './data/VGG/modVGG.pt')
