import os, sys, time, math, pickle, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import utils
from models import networks
from data import jigsaw_data_helper


parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name',  help='')
parser.add_argument('--data_path', required=False, default='data_path',  help='data path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--grid_size', type=int, default=3, help='grid size')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=64, help='input size')
parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=10, help='Number of classes for the jigsaw task')
parser.add_argument("--bias_whole_image", default=0.0, type=float, help='If set, will bias the training procedure to show more often the whole image')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--latest_generator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_discriminator_model', required=False, default='', help='the latest trained model path')
args = parser.parse_args()


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()


# results save path
if not os.path.isdir(os.path.join(args.name + '_results', 'Transfer')):
    os.makedirs(os.path.join(args.name + '_results', 'Transfer'))
if not os.path.isdir(os.path.join(args.name + '_results', 'model')):
    os.makedirs(os.path.join(args.name + '_results', 'model'))

sys.stdout = Logger(os.path.join(args.name + '_results', 'log.txt'))

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

print('cuda: %s' % torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

train_loader, test_loader, len_train_loader= jigsaw_data_helper.get_train_dataloader(args)

# network
G = networks.Transformer(args.in_ngc, args.out_ngc, args.ngf, args.nb, args.grid_size, args.jigsaw_n_classes)
if args.latest_generator_model != '':
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(args.latest_generator_model))
    else:
        # cpu mode
        G.load_state_dict(torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage))

D = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)
if args.latest_discriminator_model != '':
    if torch.cuda.is_available():
        D.load_state_dict(torch.load(args.latest_discriminator_model))
    else:
        D.load_state_dict(torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage))


G.to(device)
D.to(device)
G.train()
D.train()
print('---------- Networks initialized -------------')
utils.print_network(G)
utils.print_network(D)
print('-----------------------------------------------')

# loss
MSE_loss = nn.MSELoss().to(device)
L1_loss = nn.L1Loss().to(device)
CE_loss = nn.CrossEntropyLoss().to(device)
Bou_loss = pytorch_ssim.SSIM().to(device)

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

train_hist = {}
train_hist['Disc_loss'] = []
train_hist['Gen_loss'] = []
train_hist['Jig_loss'] = []
train_hist['Bou_loss'] = []
train_hist['total_time'] = []
print('training start!')
start_time = time.time()


flows = torch.from_numpy(utils.read_flows(args.grid_size, args.jigsaw_n_classes)).to(device)
real = torch.ones(args.batch_size, 1, math.ceil(args.input_size / 4), math.ceil(args.input_size / 4)).to(device)
fake = torch.zeros(args.batch_size, 1, math.ceil(args.input_size / 4), math.ceil(args.input_size / 4)).to(device)

for epoch in range(args.train_epoch):

    epoch_start_time = time.time()
    G_scheduler.step()
    D_scheduler.step()
    Disc_losses = []
    Gen_losses = []
    Con_losses = []
    Jig_losses = []

    for it, (data, jig_l) in enumerate(train_loader):
        width = data.shape[3]
        x = data[:, :, :, 0:int(width / 2)]
        y = data[:, :, :, int(width / 2):width]
        x = x[:, [2, 1, 0], :, :]
        y = y[:, [2, 1, 0], :, :]
        x, y = x.to(device), y.to(device)
        jig_l = jig_l.to(device)
        #print(jig_l)

        h = x.shape[2]
        w = x.shape[3]

        x_in = [None] * (args.grid_size ** 2)
        for i in range(args.grid_size):
            for j in range(args.grid_size):
                x_in[i*args.grid_size+j] = x[:, :, i*int(h/args.grid_size):(i+1)*int(h/args.grid_size), j*int(w/args.grid_size):(j+1)*int(w/args.grid_size)]

        x_in = torch.stack(x_in, 0)
        x_in = x_in[:,0,:,:,:]
        x_in = x_in.to(device)
        

        # train D
        D_optimizer.zero_grad()

        D_real = D(y)
        D_real_loss = MSE_loss(D_real, real)

        jig_logit, G_ = G(x_in, flows)
        
        D_fake = D(G_)
        D_fake_loss = MSE_loss(D_fake, fake)

        Disc_loss = D_real_loss + D_fake_loss
        Disc_losses.append(Disc_loss.item())
        train_hist['Disc_loss'].append(Disc_loss.item())
        

        # train G
        G_optimizer.zero_grad()

        jig_logit, G_ = G(x_in, flows)

        D_fake = D(G_)
        D_fake_loss = MSE_loss(D_fake, real)

        # To DO: jigsaw loss
        # Jig_loss = CE_loss(jig_logit, jig_ref)  # this line corresponds to the paper
        Jig_loss = CE_loss(jig_logit, jig_l)  # you can use this line for better performance.

        # boundary loss
        Bou_loss = Bou_loss(G_)

        Gen_loss = D_fake_loss + Jig_loss + Bou_loss

        Jig_losses.append(Jig_loss.item())
        train_hist['Jig_loss'].append(Jig_loss.item())

        Bou_losses.append(Bou_loss.item())
        train_hist['Bou_loss'].append(Bou_loss.item())

        Gen_losses.append(Gen_loss.item())
        train_hist['Gen_loss'].append(Gen_loss.item())
        
        Gen_loss.backward()
        G_optimizer.step()
        

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)
    print(
    '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Jig loss: %.3f，Bou loss: %.3f' % ((epoch), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
        torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Jig_losses)), torch.mean(torch.FloatTensor(Bou_losses))))

    if epoch % 10 == 1 or epoch == args.train_epoch - 1:
        with torch.no_grad():
            G.eval()
            for n, (x, jig_l) in enumerate(train_loader):
                x = x[:, :, :, 0:int(width / 2)]
                x = x.to(device)
                x_bgr = x[:, [2, 1, 0], :, :]
                x_bgr = x_bgr.to(device)

                x_in = [None] * (args.grid_size ** 2)
                for i in range(args.grid_size):
                    for j in range(args.grid_size):
                        x_in[i * args.grid_size + j] = x_bgr[:, :,
                                                       i * int(h / args.grid_size):(i + 1) * int(h / args.grid_size),
                                                       j * int(w / args.grid_size):(j + 1) * int(w / args.grid_size)]

                x_in = torch.stack(x_in, 0)
                x_in = x_in[:, 0, :, :, :]
                x_in = x_in.to(device)
                _, G_recon = G(x_in, flows)
                G_recon_rgb = G_recon[:, [2, 1, 0], :, :]
                result = torch.cat((x[0], G_recon_rgb[0]), 2)
                path = os.path.join(args.name + '_results', 'Transfer',
                                    str(epoch + 1) + '_epoch_' + args.name + '_train_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

                if n == 2:
                    break

            torch.save(G.state_dict(), os.path.join(args.name + '_results', 'model', 'generator_epoch_' + str(epoch+1)))
            torch.save(D.state_dict(), os.path.join(args.name + '_results', 'model', 'discriminator_epoch_' + str(epoch+1)))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish!... save training results")

torch.save(G.state_dict(), os.path.join(args.name + '_results', 'model', 'generator_param.pkl'))
torch.save(D.state_dict(), os.path.join(args.name + '_results', 'model', 'discriminator_param.pkl'))
with open(os.path.join(args.name + '_results', 'model', 'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)
