# param number calc
def model_param_number_calc(model_):
  return sum([p.numel() for p in model_.parameters() if p.requires_grad])
