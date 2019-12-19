import os
import torch
import os.path


class LoadSaveOutput():
    def __init__(self,output_dir,load_model):
        self.load_model = load_model
        self.save_dir = os.path.join(output_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_model = os.path.join(output_dir,'Model')

        if not os.path.exists(self.save_model):
            os.makedirs(self.save_model)
        self.log = open(self.save_dir+"logs.txt",'w')


    def saveModel(self,model,epoch,optimizer,loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, self.save_model+"/checkpoint_"+str(epoch)+".pth")

    def writeLog(self,text):
        self.log.write(text+'\n')

    def loadModel(self,model,optimizer,model_path=None):
        if model_path!=None:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(self.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model,optimizer,epoch,loss
