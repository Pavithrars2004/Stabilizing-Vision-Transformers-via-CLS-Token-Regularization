## Stabilizing Vision Transformers via CLS Token Regularization
# Abstract
Vision Transformers(ViTs) have achieved competitive per-
formance in the visual recognition tasks.However,their optimization re-
mains sensitive to the training instability, especially under limited data
regimes.In this paper, we introduce CLS Token Stability Regular-
ization (CLS-LSR), a lightweight regularization technique that ex-
plicitly constrains the temporal drift of the CLS token representation
during training.CLS-LSR enforces consistency across iterations, mitigat-
ing representational oscillations that degrade the convergence and gen-
eralization.Extensive experiments on CIFAR-100 using DeiT-S demon-
strate that CLS-LSR consistently improves classification accuracy by up
to 4.1% when combined with Exponential Moving Average(EMA) op-
timization. Ablation studies validate the individual and combined con-
tributions of CLS-LSR and EMA, while training stability analysis re-
veals consistently reduced optimization variance.The proposed method
is architecture-agnostic, computationally efficient and seamlessly inte-
grates into the existing ViT pipelines.
