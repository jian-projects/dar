from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from torch.autograd import Variable


project_root = '.'
os.chdir(project_root)

no_cuda = False
cuda_available = not no_cuda and torch.cuda.is_available()

BATCH_SIZE = 64
EPOCH = 100
SEED = 8

torch.manual_seed(SEED)

device = torch.device("cuda" if cuda_available else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_available else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        
        self.fc1 = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(30, 400)
        self.fc4 = nn.Linear(400, 794)
        
        self.lb = LabelBinarizer()
    #将标签进行one-hot编码
    def to_categrical(self, y: torch.FloatTensor):
        y_n = y.numpy()
        self.lb.fit(list(range(0,10)))
        y_one_hot = self.lb.transform(y_n)
        floatTensor = torch.FloatTensor(y_one_hot)
        return floatTensor
        
    def encode(self, x, y):
        y_c = self.to_categrical(y).type_as(x)
        #输入样本和标签y的one-hot向量连接
        con = torch.cat((x, y_c), 1)
        h1 = F.relu(self.fc1(con))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        #训练时使用重参数化技巧，测试时不用。（测试时应该可以用）
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, y):
        y_c = self.to_categrical(y).type_as(z)
        #解码器的输入：将z和y的one-hot向量连接
        cat = torch.cat((z, y_c), 1)
        h3 = F.relu(self.fc3(cat))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    #Sets the module in training mode.
    model.train()
    train_loss = 0

#     batch_idx, (data, label) =enumerate(train_loader).__next__()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device) #[64, 1, 28, 28]
        
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data, label)
#         print(recon_batch.shape) #[64, 794]
        #训练样本展平，在每个样本后面连接标签的one-hot向量
        flat_data = data.view(-1, data.shape[2]*data.shape[3])
#         print(data.shape, flat_data.shape)
        y_condition = model.to_categrical(label).type_as(flat_data)
        con = torch.cat((flat_data, y_condition), 1)
        
        loss = loss_function(recon_batch, con, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data, label)
            
            flat_data = data.view(-1, data.shape[2]*data.shape[3])
            
            y_condition = model.to_categrical(label)
            con = torch.cat((flat_data, y_condition), 1)
            test_loss += loss_function(recon_batch, con, mu, logvar).item()

            if i == 0:
                n = min(data.size(0), 8)
                recon_image = recon_batch[:, 0:recon_batch.shape[1]-10]
                print(recon_image.shape)
                recon_image = recon_image.view(BATCH_SIZE, 1, 28,28)
                print('---',recon_image.shape)
                comparison = torch.cat([data[:n],
                                      recon_image.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, EPOCH + 1):
    train(epoch)
    test(epoch)

    with torch.no_grad():
        #采样过程
        sample = torch.randn(64, 20).to(device)
      
        c = np.zeros(shape=(sample.shape[0],))
        rand = np.random.randint(0, 10)
        print(f"Random number: {rand}")
        c[:] = rand
        c = torch.FloatTensor(c)
        sample = model.decode(sample, c).cpu()
        #模型的输出矩阵：每一行的末尾都加了one-hot向量，要去掉这个one-hot向量再转换为图片。
        generated_image = sample[:, 0:sample.shape[1]-10]
        
        
        save_image(generated_image.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')

