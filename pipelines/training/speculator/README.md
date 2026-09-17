# Speculator

This subcategory contains pipelines in the **Speculator** group:

- [Speculator Data Only Pipeline](./data_only/README.md): Extract verifier hidden states using managed or external vLLM.
- [Speculator Offline Pipeline](./offline/README.md): Extract hidden states, then train from them in a separate task.
- [Speculator Online Pipeline](./online/README.md): Extract hidden states and train the draft model in one task.
- [Speculator Train Only Pipeline](./train_only/README.md): Train a draft model from hidden states already stored on the workspace PVC.
