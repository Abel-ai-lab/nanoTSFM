from nanotsfm.data import Windows, load

# Load the training dataset.
train_set = load("GEP-M", "train")

# Load the official statistical profiles.
train_windows = Windows(
    train_set,
    context=512,
    horizon=1,
    count=1000,
    seed=7,
    power=1.0,
)

# Load validation dataset.
validation_set = load("GEP-M", "validation")

# Construct validation windows.
validation_windows = Windows(
    validation_set,
    context=512,
    horizon=96,
    count=100,
    seed=0,
)

# Inspect training sample.
history, target = train_windows[0]

print("Training history:", history.shape)
print("Training target:", target.shape)

# Inspect validation sample.
history, target = validation_windows[0]

print("Validation history:", history.shape)
print("Validation target:", target.shape)