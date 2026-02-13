# network.py
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data / 255.0
y = mnist.target.astype(int)

rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
X, y = X[indices], y[indices]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training: {len(X_train)}, Test: {len(X_test)}")


def one_hot(labels, num_classes=10):
    result = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        result[i, label] = 1.0
    return result

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)


def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(y_hat, y_true):
    y_hat_clipped = np.clip(y_hat, 1e-12, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_hat_clipped), axis=1))


W_lin = rng.normal(0, 0.01, size=(784, 10))
b_lin = np.zeros(10)

epochs = 30
batch_size = 64
lr = 0.1

for epoch in range(epochs):
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train_oh[perm]

    for start in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[start:start + batch_size]
        y_batch = y_shuffled[start:start + batch_size]

        # Forward
        z = X_batch @ W_lin + b_lin
        y_hat = softmax(z)

        # Backward
        delta = (y_hat - y_batch) / len(X_batch)
        dW = X_batch.T @ delta
        db = delta.sum(axis=0)

        # Update
        W_lin -= lr * dW
        b_lin -= lr * db

preds_lin = (X_test @ W_lin + b_lin).argmax(axis=1)
acc_lin = (preds_lin == y_test).sum() / len(y_test)
print(f"Linear classifier accuracy: {acc_lin:.1%}")


fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for digit, ax in enumerate(axes):
    ax.imshow(W_lin[:, digit].reshape(28, 28), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(str(digit), fontsize=12)
    ax.axis("off")
plt.suptitle("What the linear classifier looks for", fontsize=14)
plt.tight_layout()
plt.savefig("linear_weights.png", dpi=150)
plt.show()


def initialize_weights(rng):
    W1 = rng.normal(0, np.sqrt(2.0 / 784), size=(784, 128))
    b1 = np.zeros(128)
    W2 = rng.normal(0, np.sqrt(2.0 / 128), size=(128, 10))
    b2 = np.zeros(10)
    return W1, b1, W2, b2

W1, b1, W2, b2 = initialize_weights(rng)
print(f"Total parameters: {W1.size + b1.size + W2.size + b2.size}")


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1            # (batch, 784) @ (784, 128) = (batch, 128)
    h = relu(z1)                 # (batch, 128)
    z2 = h @ W2 + b2            # (batch, 128) @ (128, 10) = (batch, 10)
    y_hat = softmax(z2)          # (batch, 10)
    return z1, h, z2, y_hat


def backward(X, z1, h, y_hat, y_true, W2):
    batch_size = X.shape[0]

    # Step 1: output error
    delta2 = (y_hat - y_true) / batch_size

    # Step 2: gradients for W2, b2
    dW2 = h.T @ delta2
    db2 = delta2.sum(axis=0)

    # Step 3: propagate error back through ReLU
    delta1 = (delta2 @ W2.T) * relu_derivative(z1)
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0)

    return dW1, db1, dW2, db2


def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


epochs = 30
batch_size = 64
lr = 0.1
losses = []

for epoch in range(epochs):
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train_oh[perm]

    epoch_loss = 0.0
    num_batches = 0

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        z1, h, z2, y_hat = forward(X_batch, W1, b1, W2, b2)
        loss = cross_entropy_loss(y_hat, y_batch)
        epoch_loss += loss
        num_batches += 1

        dW1, db1, dW2, db2 = backward(X_batch, z1, h, y_hat, y_batch, W2)
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")


plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training loss over time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()


def predict(X, W1, b1, W2, b2):
    _, _, _, y_hat = forward(X, W1, b1, W2, b2)
    return y_hat.argmax(axis=1)

test_preds = predict(X_test, W1, b1, W2, b2)
accuracy = (test_preds == y_test).sum() / len(y_test)
print(f"\nLinear classifier: {acc_lin:.1%}")
print(f"Neural network:    {accuracy:.1%}")


train_preds = predict(X_train, W1, b1, W2, b2)
train_accuracy = (train_preds == y_train).sum() / len(y_train)
print(f"\nTraining accuracy: {train_accuracy:.1%}")
print(f"Test accuracy:     {accuracy:.1%}")
print(f"Gap:               {train_accuracy - accuracy:.1%}")


def forward_no_relu(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = z1                       # no activation!
    z2 = h @ W2 + b2
    y_hat = softmax(z2)
    return z1, h, z2, y_hat

def backward_no_relu(X, z1, h, y_hat, y_true, W2):
    batch_size = X.shape[0]
    delta2 = (y_hat - y_true) / batch_size
    dW2 = h.T @ delta2
    db2 = delta2.sum(axis=0)
    delta1 = delta2 @ W2.T       # no ReLU gate
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0)
    return dW1, db1, dW2, db2

W1_nr, b1_nr, W2_nr, b2_nr = initialize_weights(np.random.default_rng(42))

for epoch in range(30):
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train_oh[perm]

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        z1, h, z2, y_hat = forward_no_relu(X_batch, W1_nr, b1_nr, W2_nr, b2_nr)
        dW1, db1, dW2, db2 = backward_no_relu(X_batch, z1, h, y_hat, y_batch, W2_nr)
        W1_nr, b1_nr, W2_nr, b2_nr = update_weights(
            W1_nr, b1_nr, W2_nr, b2_nr, dW1, db1, dW2, db2, lr)

preds_nr = forward_no_relu(X_test, W1_nr, b1_nr, W2_nr, b2_nr)[3].argmax(axis=1)
acc_nr = (preds_nr == y_test).sum() / len(y_test)
print(f"\nLinear classifier (no hidden layer): {acc_lin:.1%}")
print(f"Two layers, no ReLU:                 {acc_nr:.1%}")
print(f"Neural network (with ReLU):          {accuracy:.1%}")


def pca_2d(vecs):
    centered = vecs - vecs.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top2 = eigenvectors[:, -2:][:, ::-1]
    return centered @ top2


sample_idx = rng.choice(len(X_test), 3000, replace=False)
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]

z1_sample = X_sample @ W1 + b1
h_sample = relu(z1_sample)

pixel_2d = pca_2d(X_sample)
hidden_2d = pca_2d(h_sample)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for digit in range(10):
    mask = (y_sample == digit)
    ax1.scatter(pixel_2d[mask, 0], pixel_2d[mask, 1],
                c=[colors[digit]], s=5, alpha=0.5, label=str(digit))
    ax2.scatter(hidden_2d[mask, 0], hidden_2d[mask, 1],
                c=[colors[digit]], s=5, alpha=0.5, label=str(digit))

ax1.set_title("Raw pixels (PCA to 2D)", fontsize=14)
ax1.legend(markerscale=3, fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_title("After hidden layer (PCA to 2D)", fontsize=14)
ax2.legend(markerscale=3, fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("representations.png", dpi=150)
plt.show()


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix

confusion = confusion_matrix(y_test, test_preds)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(confusion, cmap="Blues")
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Neural network confusion matrix", fontsize=14)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        color = "white" if confusion[i, j] > confusion.max() / 2 else "black"
        ax.text(j, i, str(confusion[i, j]),
                ha="center", va="center", color=color, fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("nn_confusion.png", dpi=150)
plt.show()


wrong = np.where(test_preds != y_test)[0]
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    idx = wrong[i]
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"{test_preds[idx]}(={y_test[idx]})", fontsize=10)
    ax.axis("off")
plt.suptitle("Wrong predictions (predicted = actual)", fontsize=13)
plt.tight_layout()
plt.savefig("errors.png", dpi=150)
plt.show()
