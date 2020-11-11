from SiamFC import *
from dataset import *
# from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from Config import *
from Utils import *
import torchvision.transforms as transforms
from augmentation import *
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(data_dir, train_imdb, val_imdb, model_save_path="./model/", use_gpu=True):
    config = Config()
    
    # do data augmentation
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride

    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # load data 
    train_dataset = ILSVRCDataSet(train_imdb, data_dir, config, train_z_transforms, train_x_transforms)
    val_dataset = ILSVRCDataSet(val_imdb, data_dir, config, valid_z_transforms, valid_x_transforms, "Validation"))

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.val_num_workers, drop_last=True)


    # create the net, and complete forward computation

    net = SiamNet()

    if use_gpu:
        net.cuda()

    # define training stretegy:
    optimizer = torch.optim.SGD([
        {'param': net.embedding_function.parameters()},
        {'param': net.filter.bias},
        {'param': net.filter.weight, 'lr', 0},
    ], config.lr, config.momentum, config.weight_decay)

    #adjust learning in each epoch
    scheduler = StepLR(optimizer, config.step_size, config.gamma)

    # Notice: in Utils.py, we have known that 
    # the label for each train is same!
    # cause all the target is removed to the center,
    # so the distance between center and target is all the same!

    train_response_flag = False
    valid_response_flag = False

    for i in range(config.num_epoch):
        # set learning rate step strategy
        scheduler.step()
        # set the net to the mode train
        net.train()
        train_loss = []

        for j, data in enumerate(tqdm(train_loader)):

            exemplar_imgs, instance_imgs = data

            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            
            output = net.forward(exemplar_imgs, instance_imgs)

            if not train_response_flag:
                train_response_flag = True
                respose_size = output.size[2:4]
                train_label, train_weight = create_label(respose_size, config, use_gpu)
            # set gradient to zero!
            optimizer.zero_grad()
            # loss
            loss = net.weight_loss(output, train_label, train_weight)
            # backward
            loss.backword()
            # update parameter
            optimizer.step()
            train_loss.append(loss.data)
        
        # save model
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(net, model_save_path + "SiamFc_" + str(i + 1) + "_model.pth")

        # validation
        net.eval()

        val_loss = []

        for j, data in enumerate(tqdm(val_loader)):

            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            # create label for validation (only do it one time)
            if not valid_response_flag:
                valid_response_flag = True
                response_size = output.shape[2:4]
                valid_eltwise_label, valid_instance_weight = create_label(response_size, config, use_gpu)

            # loss
            loss = net.weight_loss(output, valid_eltwise_label, valid_instance_weight)

            # collect validation loss
            val_loss.append(loss.data)

        print ("Epoch %d   training loss: %f, validation loss: %f" % (i+1, np.mean(train_loss), np.mean(val_loss)))

if __name__ == "__main__":
    data_dir = ''
    train_imdb = ''
    val_imdb = ''

    train(data_dir, train_imdb, val_imdb)