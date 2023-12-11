import torch, datetime
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        out_features = list(self.backbone.modules())[-1].out_features

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        
        combined_features = feat1 * feat2
        
        output = self.cls_head(combined_features)
       
        return output
    
def train_siamese(model, device, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.BCELoss()
    for batch_idx, batch in enumerate(train_loader):
        batch['img_1'] = batch['img_1'].to(device)
        batch['img_2'] = batch['img_2'].to(device)
        batch['target'] = batch['target'].to(device)
       
        outputs = model(batch['img_1'], batch['img_2']).squeeze()
        
        loss = criterion(outputs, batch['target'])
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(batch['img_1']), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test_siamese(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch['img_1'] = batch['img_1'].to(device)
            batch['img_2'] = batch['img_2'].to(device)
            batch['target'] = batch['target'].to(device)
            outputs = model(batch['img_1'], batch['img_2']).squeeze(0)
            test_loss += criterion(outputs, batch['target']).sum().item() 
            pred = torch.where(outputs > 0.5, 1, 0)  
            correct += pred.eq(batch['target'].view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct