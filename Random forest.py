import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

def show_mnist_images(images, labels):
    images = images.numpy()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    plt.show()


data_iter = iter(train_loader)
images, labels = next(data_iter)
show_mnist_images(images, labels)


def prepare_data(loader):
    data = []
    targets = []
    for images, labels in loader:
        images = images.view(images.size(0), -1)
        data.append(images.numpy())
        targets.append(labels.numpy())
    data = np.vstack(data)
    targets = np.hstack(targets)
    return data, targets

X_train, y_train = prepare_data(train_loader)
X_test, y_test = prepare_data(test_loader)


print("Start 5-fold cross validation...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)


cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=kfold, scoring='accuracy')
print("5-fold cross validation accuracy:", cv_scores)
print("Average accuracy：", np.mean(cv_scores))


print("Start training the random forest model...")
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)


print("Test set classification results:")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Test set accuracy: {accuracy:.4f}")
print("Classification Report:\n", class_report)


plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

