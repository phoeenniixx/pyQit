import pennylane as qml
from pennylane import numpy as np

class Trainer:
    def __init__(self, backend_type="pennylane", max_epochs=10, learning_rate=0.01):
        self.backend_type = backend_type
        self.max_epochs = max_epochs
        self.lr = learning_rate

    def fit(self, model, X, y, optimizer=None, loss_fn=None):
        if self.backend_type == "pennylane":
            self._fit_pennylane(model, X, y, optimizer, loss_fn)
        elif self.backend_type == "torch":
            self._fit_torch(model, X, y, optimizer, loss_fn)


    def _fit_pennylane(self, model, X, y, optimizer=None, loss_fn=None, batch_size=32):
        if optimizer is None:
            opt = qml.AdamOptimizer(stepsize=self.lr)
        else:
            opt = optimizer

        if loss_fn is None:
            loss_fn = lambda p, t: np.mean((p - t) ** 2)

        weight_keys = list(model.weights.keys())
        current_weights = [
            np.array(model.weights[k], requires_grad=True) for k in weight_keys
        ]

        X = np.array(X, requires_grad=False)
        y = np.array(y, requires_grad=False)
        n_samples = len(X)

        print(f"Starting Pure PennyLane Training on {self.backend_type}...")
        
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                def batch_cost_fn(*weight_tensors):
                    preds = model.forward_from_tensors(X_batch, *weight_tensors)
                    return loss_fn(preds, y_batch)
                
                grad_fn = qml.grad(batch_cost_fn)
                gradients = grad_fn(*current_weights)
                
                batch_loss = batch_cost_fn(*current_weights)
                
                updated_weights = []
                for w, g in zip(current_weights, gradients):
                    new_w = w - self.lr * g
                    new_w = np.array(new_w, requires_grad=True)
                    updated_weights.append(new_w)
                
                current_weights = updated_weights
                
                epoch_loss += batch_loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1:03d}/{self.max_epochs} | Avg Loss: {avg_loss:.6f}")

        final_weight_dict = {k: w for k, w in zip(weight_keys, current_weights)}
        model.update_weights(final_weight_dict)
        print("Training Complete.")

    def _fit_torch(self, model, train_loader, val_loader=None, optimizer=None, loss_fn=None):
        pass



