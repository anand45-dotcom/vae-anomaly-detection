Variational Autoencoder (VAE) for High-Dimensional Anomaly Detection
🔹 Architecture Design Rationale
The VAE architecture was designed to balance expressive reconstruction capability with latent regularization. The encoder maps 100-dimensional inputs into a lower-dimensional latent space (tested latent_dim ∈ {2, 4, 8}). The decoder reconstructs inputs from sampled latent representations using the reparameterization trick.
The loss function combines:
Reconstruction Loss (MSE): Encourages accurate reconstruction of normal data.
KL Divergence: Regularizes the latent distribution toward a standard normal prior, preventing overfitting and enforcing structure.
The beta coefficient controls the trade-off:
Higher β → stronger regularization, better latent structure, but poorer reconstruction.
Lower β → better reconstruction but weaker anomaly separability.
🔹 Hyperparameter Search Analysis
A structured grid search (3×3) explored:
Latent dimensions: {2, 4, 8}
Beta values: {0.5, 1.0, 2.0}
Evaluation metric: F1-score on validation set.
Observation:
The configuration with latent_dim=4 and β=1.0 produced the best trade-off between reconstruction fidelity and anomaly separability.
However, the optimized F1-score (0.667) was slightly lower than the baseline (0.678), indicating that increased latent regularization reduced reconstruction contrast between normal and anomalous samples.
This suggests the baseline model was already near-optimal for the synthetic anomaly distribution.
🔹 Threshold Selection Justification
The anomaly threshold was selected as the 98th percentile of training reconstruction errors.
Rationale:
Only normal data was used during training.
The top 2% highest reconstruction errors approximate extreme deviations.
This statistically aligns with the injected 2% anomaly ratio.
Using percentile-based thresholding ensures a distribution-aware decision boundary rather than arbitrary cutoff.
Future improvements may include:
Extreme Value Theory (EVT)
ROC-based threshold optimization
Adaptive thresholding via validation data
🔹 Trade-off Between Reconstruction and KL Regularization
Increasing β strengthens latent regularization but reduces reconstruction sharpness.
Empirical observation showed:
Higher β values compressed latent space too aggressively, reducing the reconstruction gap between normal and anomalous data. This reduced anomaly detection sensitivity.
Therefore, moderate β values achieved better balance.
🔹 Critical Reflection
The slight decrease in optimized F1-score suggests:
The synthetic anomaly separation may already be linearly separable.
Excess regularization reduced useful variance.
The search space was limited (3×3 grid).
More advanced strategies such as:
Beta scheduling
Larger latent search
Early stopping
Bayesian optimization
could further improve performance.
