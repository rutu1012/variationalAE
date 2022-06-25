import torch
import torch
from architecture.architecture import vae
from trainNtest.trainNtest import training, testing
from utility.utility import cost_graph,view_images


def main(training_data,testing_data,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = vae().to(device)
    opt = model.optimizer()
    train_loss = []
    test_loss = []
    train_output = []
    test_output =[]
    print("Training Model")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        train_epoch_loss = training(model, training_data, opt,  epoch, train_output, device)
        print(f"Train Loss: {train_epoch_loss}")
        train_loss.append(train_epoch_loss)
    print("Testing Model")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        test_epoch_loss = testing(model, testing_data, epoch, test_output, device)
        print(f"Test Loss: {test_epoch_loss}")
        test_loss.append(test_epoch_loss)
    
    # torch.save(model.state_dict(), FILE)  # saves the trained model at the specified path
    # files.download('VAE_model.pth')

    cost_graph(train_loss,"Train Loss")
    cost_graph(test_loss,"Test Loss") 
    
    # plot_latent(model,train_data)
    print("Train result") 
    view_images(train_output,num_epochs)
    print("Test result")
    view_images(test_output,num_epochs)
