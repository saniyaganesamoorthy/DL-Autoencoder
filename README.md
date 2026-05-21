# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

This code implements a Denoising Autoencoder using PyTorch to clean noisy images from the MNIST dataset. It uses a convolutional neural network architecture, where the encoder compresses the input image into a lower-dimensional representation, and the decoder reconstructs the original image from this compressed form. To train the model to remove noise, Gaussian noise is added to the clean images, and the network learns to recover the original from the noisy version. The training process uses Mean Squared Error (MSE) as the loss function to measure the reconstruction error and the Adam optimizer to update the model weights. The autoencoder is trained over multiple epochs using mini-batches of data for efficiency. After training, the model's performance is visually evaluated by displaying the original, noisy, and denoised images side by side.

## DESIGN STEPS

STEP 1:
Problem Understanding and Dataset Selection

STEP 2:
Preprocessing the Dataset

STEP 3:
Design the Convolutional Autoencoder Architecture

STEP 4:
Compile and Train the Model

STEP 5:
Evaluate the Model

STEP 6:
Visualization and Analysis


## PROGRAM

### Name: SANIYA G

### Register Number: 212223240147

```python

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Name:SANIYA G")
print("Register Number:212223240147")
summary(model, input_size=(1, 28, 28))

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name:SANIYA G")
    print("Register Number:212223240147")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")


```

### OUTPUT

### Model Summary
<img width="468" height="340" alt="image" src="https://github.com/user-attachments/assets/be3fbfb2-cc79-41d5-acd4-10ee9ef6d979" />

### Training loss
<img width="289" height="102" alt="image" src="https://github.com/user-attachments/assets/5e51c48e-a24a-446c-9be9-aaffbc16d0f3" />


## Original vs Noisy Vs Reconstructed Image
<img width="807" height="303" alt="image" src="https://github.com/user-attachments/assets/97cbc97a-d76a-469b-9668-162dd7f1b108" />


## RESULT

Thus, develop a convolutional autoencoder for image denoising application excuted succesfully

