import torch
import torch.nn as nn
from torch.nn.functional import softmax
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torch.optim as optim
from torchvision import transforms
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from skimage import io, transform
from one_hot_try import covertToOnehot
from one_hot_try import genRefMap

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


""" TRAINING DATA PREPARATION """

# Mode parameters
train = True
by_category = True     # Load data from selected categories

# Root directory for dataset

ref_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/ADEChallengeData2016/"
anno_root_dir = "/Users/ericfu/Documents/ANU_Master/COMP8755_Project/dataset/" \
               "ADEChallengeData2016/annotations/training/"

category = ['beach']      # multiple categories stored in a list
ref_list_name = "sceneCategories.txt"
file_list = []         # segmentation maps to use as training examples

ref_list = os.path.join(ref_root_dir, ref_list_name)
f = open(ref_list)

classSet = set()    # what classes that the dataset contains.

# Prepare segmentation maps from specified category of scenes ！！
if by_category:

    line = f.readline()
    while line:
        n, c = line.split(" ")
        c = c[:-1]
        path = os.path.join(anno_root_dir, n + ".png")
        if c in category:
            if train:
                if 'train' in n and io.imread(path).shape == (256 ,256):
                    file_list.append(n + ".png")
            else:
                if "val" in n and io.imread(path).shape == (256 ,256):
                    file_list.append(n + ".png")

        line = f.readline()
    f.close()

if by_category is False:

    line = f.readline()
    while line:
        n, c = line.split(" ")
        if train:
            if 'train' in n:
                file_list.append(n + ".png")
        else:
            if "val" in n:
                file_list.append(n + ".png")
        line = f.readline()
    f.close()

# Prepare the class list.
for file in file_list:
    im = io.imread(os.path.join(anno_root_dir, file))
    imArray = np.asarray(im).reshape(1, -1)
    imClasses = np.unique(imArray)
    for c in imClasses:
        classSet.add(c)

refMap, num_classes = genRefMap(classSet)



""" TRAINING PARAMETERS """

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 4

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images.
nc = num_classes

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 32

# Size of feature maps in discriminator (SET TO WHAT NUMBER????)
ndf = 32

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1



""" PREPARE THE DATASET """

class SegMapDataset (Dataset):
    """ Segmentation map dataset for 1st phase of the network. """

    def __init__(self, file_list, anno_root_dir, transform=None):
        """
        Args:
            :param file_list: list of selected image file names to retrieve
            :param anno_root_dir: Directory that the samples are stored
            :param transform: Optional transform to be applied
            on a sample.
        """
        self.file_list = file_list
        self.anno_root_dir = anno_root_dir
        self.transform = transform

    def __len__(self):
        return len(file_list)

    def __getitem__(self, idx):
        """
        :param idx: line index of the file in the file list
        :return: one-hot encoded segmentation map in shape of 256*256-by-number_of_semantic_categories
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        seg_map_name = os.path.join(self.anno_root_dir, self.file_list[idx])
        image = io.imread(seg_map_name)
        img = covertToOnehot(image, refMap, num_classes)

        sample = {'image': img, 'fileName': seg_map_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size. Always assume that we want
    a square image input (h=w)

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):      # 为了将一个类作为函数调用
        image, fileName = sample['image'], sample['fileName']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            # if h > w:
            #     new_h, new_w = self.output_size * h / w, self.output_size
            # else:
            #     new_h, new_w = self.output_size, self.output_size * w / h
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), preserve_range=True)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img, 'fileName': fileName}


dataset = SegMapDataset(file_list=file_list, anno_root_dir=anno_root_dir)

dataloader = DataLoader(dataset, batch_size=batch_size,               # load data in chosen manner
                            shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# # Plot some training images
# real_batch = next(iter(dataloader))
# print(real_batch)
# plt.figure(figsize=(2,4))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch['image'][0].to(device)[:8], padding=2, normalize=False).cpu(),(1,2,0)))
# plt.imshow(dataset[0]['image'])
# plt.pause(10)


""" GAN IMPLEMENTATION """

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Softmax(dim=1)

            # state size. (nc) x 256 x 256

        )

    def forward(self, input):
        return softmax(self.main(input), 2)

    # Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = torch.tensor(input, dtype=torch.float32)
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

## LOSS FUNCTION ##
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(32, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        print('Epoch: ', epoch, 'batch: ', i)
        print('A BATCH')
        print(data['image'].shape)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data['image'].to(device)
        print("real cpu shape: ", real_cpu.shape)
        b_size = real_cpu.size(0)
        print ('b_size: ', b_size)
        label = torch.full((b_size,), real_label, device=device)
        print("label shape: ", label.shape)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        print("output:", output)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)*100
        # Generate fake image batch with G
        fake = netG(noise)
        print("fake size: ", fake.shape)
        print(fake[2][10])
        label.fill_(fake_label)
        print("label fake: ", label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        print("fake output: ", output)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        print("error D real: ", errD_real)
        print("error D fake: ", errD_fake)
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        print("G output: ", output.shape)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_noise).detach().cpu()
        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        #
        # iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
