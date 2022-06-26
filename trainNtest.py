import torch
from tqdm import tqdm

# num_epochs = 25


def training(model, dataloader, opt, epoch, train_output, latent_space, device, num_epochs):
    print('Training')
    # costs=[]
    model.train()
    running_loss = 0.0
    for (img, class_name) in tqdm(dataloader):
        img = img.to(device)
        # img = img.view(img.size(0), -1)
        # print("IN: ",img[0].shape)
        opt.zero_grad()
        # forward
        out, mean, sigma, sample = model(img)

        out = out.to(device)
        loss = model.criterion(img, out, mean, sigma)
        if epoch == num_epochs - 1:
            latent_space.append([sample, class_name])
        # loss.sum().backward()  # why its given that loss is partially computer in each gpu so we have to do this
        loss.backward()
        # print(loss)
        running_loss += loss.item()
        opt.step()

    train_output.append((epoch, img, out))

    return running_loss / len(dataloader), latent_space


def testing(model, dataloader, epoch, test_output, device):
    print('Testing')
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (img, _) in tqdm(dataloader):
            img = img.to(device)
            # img = img.view(img.size(0), -1)
            out, mean, sigma, sample = model(img)

            out = out.to(device)
            # viewImage(out,"test",epoch)

            loss = model.criterion(img, out, mean, sigma)

            running_loss += loss.item()

        test_output.append((epoch, img, out))

    return running_loss / len(dataloader)
