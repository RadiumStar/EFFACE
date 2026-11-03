# Logistic Regression Toy Example 

We consider a binary classification task on digits 3 and 8 from the MNIST dataset, using logistic regression with $L_2$ regularization

$$
\frac{1}{n} \sum_{j=1}^{n} \log(1 + \exp(-y_j \theta^\top x_j)) + \frac{\lambda_{\mathrm{reg}}}{2} \|\theta\|_{2}^2,
$$

as the loss function. Here $n$ is the total number of data samples, $x_j \in \mathbb{R}^d$ is the flattened feature vector of the $j$-th data sample with $d = 28 \times 28 = 784$, $y_j \in \{-1, 1\}$ is the label of the $j$-th data sample, and $\lambda_{\mathrm{reg}}$ is the regularization parameter and set to $0.001$ by default. We partition the data samples into 10 clients. We select 5\% of each local dataset $\mathcal{D}_i$ as the local forgetting dataset $\mathcal{D}_i^f$. The selected data samples are all originally labeled as $-1$ (digit 3). We inject backdoor triggers into them, and change their labels to target $1$ (digit 8).

- To obtain the original model $\theta^0$, run the following command:
```bash
cd LogisticRegression
bash scripts/pretrain.sh
```

- To unlearn the backdoor triggers, run the following command:
```bash
bash scripts/unlearn.sh
```

- To visualize the trajectory of the global model, run the following command:
```bash
python visual_trajectory.py --alpha 1e8
```

where `alpha` is the parameter for Dirichlet distribution in the data partitioning. The default is `1e8`, which corresponds to the i.i.d. setting. We visualize the trajectory under i.i.d. setting since the data heterogeneity can introduce additional noise to the trajectory of global models, making it harder to visualize the differences between communicators and disclose the dilemmas of current error feedback mechanisms in federated unlearning. 