
from LoadData  import DatasetLoader
from LoadSaveModel import LoadSaveOutput
from ResidualDenseNetwork import  ResidualDenseNetwork
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision import models, transforms
from torch.nn import init
import numpy


class Controller():
    def __init__(self):
        '''
        self.folder_path = '/content/drive/My Drive/670project/'
        self.target_train = 'dataset/DIV2K_train_HR/'
        self.source_train = "dataset/DIV2K_train_LR_bicubic_X4/"
        self.target_val = 'dataset/DIV2K_valid_HR/'
        self.source_val = "dataset/DIV2K_valid_LR_bicubic_X4/"
        self.source_test =""
        self.target_test = ""
        self.load_model = ""
        self.output_dir = "/content/drive/My Drive/670project/output/"
        '''
        self.folder_path = '/mnt/nfs/scratch1/mnabail/'
        self.target_train = 'dataset670/DIV2K_train_HR/'
        self.source_train = "dataset670/DIV2K_train_LR_bicubic_X4/"
        self.target_val = 'dataset670/DIV2K_valid_HR/'
        self.source_val = "dataset670/DIV2K_valid_LR_bicubic_X4/"
        self.source_test =""
        self.target_test = ""
        self.load_model = ""
        self.output_dir = "/mnt/nfs/scratch1/mnabail/output670/"
        self.batch_size = 6
        self.batch_size_val = 4
        self.epochs =40
        self.scale = 2
        self.learning_rate = 1e-4
        self.scale = 4
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def loadDataset(self,type):
        if type=="train":
            dataset = DatasetLoader(self.folder_path,self.source_train,self.target_train,self.scale)
            torch_loader = torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        elif type=="val":
            dataset_val = DatasetLoader(self.folder_path,self.source_val,self.target_val,self.scale)
            torch_loader = torch.utils.data.DataLoader(dataset_val,batch_size=self.batch_size_val,shuffle=True)
        elif type=="test":
            dataset = DatasetLoader(self.folder_path,self.source_test,self.target_test,self.scale)
            torch_loader = torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        
        return torch_loader

    def update_lr(self,optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init.kaiming_normal(m.weight.data)
    def train_model(self,train_loader,val_loader=None):
        lr =self.learning_rate
        epoch_loss = []
        model = ResidualDenseNetwork(self.scale)
        model.apply(self.weights_init_kaiming)
        model = model.to(self.device)
        #Initialize loss function
        criterion = nn.MSELoss()
        #Initialize the optimizer to be used
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        #Train for the no of epochs
        for epoch in range(self.epochs):
            print("Staring epoch:",epoch)

            for i,(lr_img,hr_img) in enumerate(train_loader):
                #continue
                #print("aug lr:", lr_img.shape, " aug hr:", hr_img.shape)
                #print("self.device:",self.device)
                hr_img = hr_img.to(dtype=torch.float) / float(255)
                lr_img = lr_img.to(dtype=torch.float) / float(255)
                #print(hr_img[0].data)
                if i%50==0:
                    print("Batch :",i)
                hr_img = Variable(hr_img.to(self.device))
                lr_img = Variable(lr_img.to(self.device))
                optimizer.zero_grad()
                pred_hr = model(lr_img)
                loss = criterion(pred_hr,hr_img)
                #Backward pass
                loss.backward()
                optimizer.step()
                #print(pred_hr[0].data)
                #imgplot = plt.imshow(pred_hr[0].data.numpy())
                #plt.show()
                if (i + 1) % 100 == 0:
                    log_v ="Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, self.epochs, i + 1, len(train_loader), loss.item())
                    self.output.writeLog(log_v)

            val_loss= 0
            for j,(lr_img_val,hr_img_val) in enumerate(val_loader):
                #print("aug lr:", lr_img.shape, " aug hr:", hr_img.shape)
                #print("")
                hr_img_val = hr_img.to(dtype=torch.float) / float(255)
                lr_img_val = lr_img.to(dtype=torch.float) / float(255)
                hr_img_val = Variable(hr_img_val.to(self.device))
                lr_img_val = Variable(lr_img_val.to(self.device))
                optimizer.zero_grad()
                #print("start val")
                pred_hr_val = model(lr_img_val)
                #print("end val")
                loss_val = criterion(pred_hr_val,hr_img_val)
                #print(loss_val.item())
                val_loss+=loss_val.item()
                loss_val =None
                pred_hr_val = None
                
                #epoch_loss.append(loss_val.item())
            print("Epoch: ",str(epoch)," validation Loss :",str(val_loss/100))
            self.output.saveModel(model,epoch,optimizer,val_loss)
            if epoch%15==0 and epoch>1:
                lr= lr/2
                self.update_lr(optimizer, lr)
            optimizer.zero_grad()
    def main(self):
        self.output = LoadSaveOutput(self.output_dir,self.load_model)
        train_loader = self.loadDataset("train")
        val_loader = self.loadDataset("val")
        self.train_model(train_loader,val_loader)
        #test_loader = self.loadDataset("test")


control = Controller()
control.main()

