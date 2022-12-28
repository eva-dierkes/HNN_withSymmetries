import torch

def my_mse_loss(model, x, y, status):
    y_hat = model.output(x)
    l_mse = torch.nn.MSELoss()(y_hat,y)

    if status != 'after':
        model.log(f'{status}/loss',l_mse, prog_bar=False, on_epoch=True, on_step=False)
        model.log(f'{status}/hamilton_loss',l_mse, prog_bar=False, on_epoch=True, on_step=False)
    return l_mse,
