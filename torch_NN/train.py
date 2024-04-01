def train_epoch(loss_function, optimizer, model, loader,test_data):
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



def train_model(loss_function, optimizer, model, loader,test_data,epochs=25):
  for i in range(epochs):
    print(f"-----------------------Epoch: {i}----------------------------------")

    train_epoch(loss_function, optimizer, model, loader,test_data)