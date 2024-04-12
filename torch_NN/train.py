import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_epoch(loss_function, optimizer, model, loader,train_data,test_data):
  for(i, (x, y)) in enumerate(loader):
    # Clear the gradients
    optimizer.zero_grad()
    # Run a forward pass
    outputs = model.forward(x)
    # Compute the batch loss
    loss = loss_function(outputs,y)
    # Calculate the gradients
    loss.backward()
    # Update the parameteres
    optimizer.step()

    if i%100 == 0:
      print(f"Batch: {i},train loss is: {loss}")
      with torch.no_grad():
        test_outputs = model.forward(test_data[0])
        test_loss = loss_function(test_outputs,test_data[1])
        print(f"test loss is {test_loss}")

  with torch.no_grad():
    train_loss = loss_function(model(train_data[0]),train_data[1])
    test_loss = loss_function(model(test_data[0]),test_data[1])

  return (train_loss,test_loss)


def train_model(loss_function, optimizer, model, loader,train_data,test_data,epochs=25):
  train_loss_list = []
  test_loss_list = []
  for i in range(epochs):
    print(f"-----------------------Epoch: {i}----------------------------------")

    loss = train_epoch(loss_function, optimizer, model, loader,train_data,test_data)
    train_loss_list.append(loss[0].detach().cpu().numpy())
    test_loss_list.append(loss[1].detach().cpu().numpy())
  
  plt.plot(list(range(len(train_loss_list))),train_loss_list,label="trian_loss")
  plt.plot(list(range(len(test_loss_list))),test_loss_list,label = "test_loss")

  plt.xlabel("epoch")
  plt.ylabel("mean_loss")
  plt.legend()
  plt.show()