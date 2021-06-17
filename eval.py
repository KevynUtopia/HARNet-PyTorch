def eval(eval_model, dataset):
      cuda = torch.cuda.is_available()
  test_model = eval_model['model']
  eval_mse, eval_psnr, eval_ssim, eval_loss = 0.0, 0.0, 0.0, 0.0
  with torch.no_grad():
    for i, data in enumerate(dataset):
      torch.cuda.empty_cache()
      x, y = data
      X, Y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
      if cuda:
        X = X.cuda()
        Y = Y.cuda()
      out = test_model(X)
      mse = Loss(out, Y)
      eval_mse += mse
      psnr = 10 * math.log10(255 * 255 / mse)
      eval_psnr += psnr
      ssim = 1-ssim_loss(out, Y)
      eval_ssim += ssim
      loss = mse + (1 - ssim)
      eval_loss += loss
      torch.cuda.empty_cache()
  print("Evaluation:")
  print("MSE: %.5f, PSNR: %.5f, SSIM: %.5f, LOSS: %.5f" %(eval_mse/90, eval_psnr/90, 1-eval_ssim/90, eval_loss/90))