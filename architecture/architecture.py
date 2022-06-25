import torch
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
learning_rate =0.001

class vae(nn.Module):
    def __init__(self):
        super(vae, self).__init__()     
                         
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, stride = 1, padding = 1), #14
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride = 2, padding = 1), #14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride = 1, padding = 1), #14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64 , 3, stride = 2, padding = 1), #7
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128, 7),  #1 x 128
            nn.Flatten(),
            nn.ReLU()
            
        )
        self.FC1 = nn.Linear(128,16) # 16x1
        self.FC2 = nn.Linear(128,16)

            
        self.decoder = nn.Sequential(   
            nn.Linear(16,128),
            nn.ReLU(),
            nn.Unflatten(1,(-1,1,1)),
            nn.ConvTranspose2d(128, 64, 7),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3, stride = 2, padding = 1, output_padding =1 ),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 4, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 1, 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )
    
    def sampling(self, mean, log_var):
        sigma  = torch.exp(0.5 * log_var)
        eps = torch.distributions.Normal(0, 1).sample(sigma.shape).to(device)
        sample = mean + torch.mul(eps,sigma)
        # eps = torch.randn_like(sigma)
        # sample = mean + eps*sigma 
        return sigma, sample  
        
    def optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(),lr =learning_rate,weight_decay=1e-5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        return optimizer
        
    def forward(self, x): # output idhar ka negative mil rha he
        x = self.encoder(x)
        # print(x.shape)
        mean = self.FC1(x)
        log_var = self.FC2(x)
        sigma, sample = self.sampling( mean, log_var)
        output = self.decoder(sample)
        return output, mean, sigma, sample
        #mic hchalu hai
    @staticmethod
    def data_fidelity(X,X_hat,eps):
        
        data_fidelity = torch.sum(X * torch.log(eps + X_hat) + (1 - X) * torch.log(eps + 1 - X_hat),axis = 1)
        data_fidelity = torch.mean(data_fidelity)
        # print("Data-fidelity: ", data_fidelity) 
        return data_fidelity
    @staticmethod
    def kl_divergence(mean,sigma):
        kl_divergence = (1/2)*torch.sum(torch.exp(sigma)+torch.square(mean)-1-sigma,axis = 1)
        kl_divergence = torch.mean(kl_divergence)
        
        # print(kl_divergence)
        # kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_divergence
        
    @staticmethod
    def criterion(X,X_hat,mean,sigma):  
        # cri=nn.BCELoss()
        # data_fidelity_loss = cri(X_hat,X)  
        data_fidelity_loss = torch.abs(vae.data_fidelity(X,X_hat,1e-10))
        kl_divergence_loss = vae.kl_divergence(mean,sigma)
        elbo_loss = data_fidelity_loss + kl_divergence_loss

        losses=[]
        losses.append((data_fidelity_loss,kl_divergence_loss,elbo_loss))
        return elbo_loss
