{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.13","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"gpu","dataSources":[{"sourceId":242592,"sourceType":"datasetVersion","datasetId":102285}],"dockerImageVersionId":30716,"isInternetEnabled":true,"language":"python","sourceType":"notebook","isGpuEnabled":true}},"nbformat_minor":4,"nbformat":4,"cells":[{"source":"<a href=\"https://www.kaggle.com/code/drkaggle22/digit-recognizer-solution-99-accuracy?scriptVersionId=187281723\" target=\"_blank\"><img align=\"left\" alt=\"Kaggle\" title=\"Open in Kaggle\" src=\"https://kaggle.com/static/images/open-in-kaggle.svg\"></a>","metadata":{},"cell_type":"markdown"},{"cell_type":"code","source":"# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only \"../input/\" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session","metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","execution":{"iopub.status.busy":"2024-06-04T19:38:02.740524Z","iopub.execute_input":"2024-06-04T19:38:02.741237Z","iopub.status.idle":"2024-06-04T19:38:02.76131Z","shell.execute_reply.started":"2024-06-04T19:38:02.741209Z","shell.execute_reply":"2024-06-04T19:38:02.760411Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"# Struct Library to Unpack Image Data","metadata":{}},{"cell_type":"code","source":"import struct\n\ndef read_idx(filename):\n    # Open the file in binary mode for reading\n    with open(filename, 'rb') as f:\n        # Read the first 4 bytes and unpack them\n        # '>HBB' means: \n        # '>' - big-endian\n        # 'H' - unsigned short (2 bytes)\n        # 'B' - unsigned byte (1 byte)\n        zero, data_type, dims = struct.unpack('>HBB', f.read(4))\n        \n        # Read the dimensions of the data\n        # '>I' means: \n        # '>' - big-endian\n        # 'I' - unsigned int (4 bytes)\n        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))\n        \n        # Read the remaining bytes and interpret them as unsigned 8-bit integers (uint8)\n        # Reshape the flat array into the shape specified by the header\n        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)\n","metadata":{"execution":{"iopub.status.busy":"2024-06-04T19:38:07.163321Z","iopub.execute_input":"2024-06-04T19:38:07.164182Z","iopub.status.idle":"2024-06-04T19:38:07.170363Z","shell.execute_reply.started":"2024-06-04T19:38:07.16415Z","shell.execute_reply":"2024-06-04T19:38:07.169424Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"## Load the Dataset","metadata":{}},{"cell_type":"code","source":"def load_mnist(image_path, label_path):\n    images = read_idx(image_path)\n    labels = read_idx(label_path)\n    return images, labels\n\ntrain_image_path = '/kaggle/input/mnist-dataset/train-images-idx3-ubyte/train-images-idx3-ubyte'\ntrain_label_path = '/kaggle/input/mnist-dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte'\ntest_image_path =  '/kaggle/input/mnist-dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'\ntest_label_path =  '/kaggle/input/mnist-dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'","metadata":{"execution":{"iopub.status.busy":"2024-06-04T19:38:08.250259Z","iopub.execute_input":"2024-06-04T19:38:08.250972Z","iopub.status.idle":"2024-06-04T19:38:08.255784Z","shell.execute_reply.started":"2024-06-04T19:38:08.250943Z","shell.execute_reply":"2024-06-04T19:38:08.254864Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"train_images, train_labels = load_mnist(train_image_path, train_label_path)\ntest_images, test_labels = load_mnist(test_image_path, test_label_path)\n\n# print the shapes\nprint(f'Train images shape: {train_images.shape}')\nprint(f'Train labels shape: {train_labels.shape}')\nprint(f'Test images shape: {test_images.shape}')\nprint(f'Test labels shape: {test_labels.shape}')","metadata":{"execution":{"iopub.status.busy":"2024-06-04T19:38:09.460281Z","iopub.execute_input":"2024-06-04T19:38:09.460652Z","iopub.status.idle":"2024-06-04T19:38:09.551628Z","shell.execute_reply.started":"2024-06-04T19:38:09.460623Z","shell.execute_reply":"2024-06-04T19:38:09.550732Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"# Reshae the arrays\ntrain_images_flat = train_images.reshape(train_images.shape[0], -1)\ntest_images_flat = test_images.reshape(test_images.shape[0], -1)\n\nX_train = train_images_flat\ny_train = train_labels\nX_test = test_images_flat\ny_test = test_labels","metadata":{"execution":{"iopub.status.busy":"2024-06-04T19:38:09.897506Z","iopub.execute_input":"2024-06-04T19:38:09.897901Z","iopub.status.idle":"2024-06-04T19:38:09.90641Z","shell.execute_reply.started":"2024-06-04T19:38:09.897873Z","shell.execute_reply":"2024-06-04T19:38:09.905351Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"from collections import Counter\n\n# Count the occurrences of each label in the y_train array using Counter\nlabel_counts = Counter(y_train)\n\n# Iterate over the items in the label_counts dictionary\nfor label, count in label_counts.items():\n    # Print out the label and its corresponding count in a formatted string\n    print(f\"Label {label}: Count {count}\")\n","metadata":{"execution":{"iopub.status.busy":"2024-06-04T19:39:24.790693Z","iopub.execute_input":"2024-06-04T19:39:24.791762Z","iopub.status.idle":"2024-06-04T19:39:24.816789Z","shell.execute_reply.started":"2024-06-04T19:39:24.791715Z","shell.execute_reply":"2024-06-04T19:39:24.816009Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"## Disribution of samples in Training set","metadata":{}},{"cell_type":"code","source":"import matplotlib.pyplot as plt\n\n# Get the unique labels and their counts\nunique_labels, counts = np.unique(train_labels, return_counts=True)\n\n# Create a bar chart\nplt.figure(figsize=(15, 10))\nbars = plt.bar(unique_labels, counts, color='skyblue')\nplt.xlabel('Labels')\nplt.ylabel('Number of Samples')\nplt.title('Distribution of Labels in MNIST Training Set')\nplt.xticks(unique_labels)\nplt.grid(axis='y', linestyle='--', alpha=0.7)\n\n# Annotate the bars with labels (digits 0-9)\nfor bar, label in zip(bars, unique_labels):\n    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 500, \n             f'{label}', ha='center', va='bottom', color='black', fontweight='bold')\n\n# Show the plot\nplt.show()","metadata":{"execution":{"iopub.status.busy":"2024-06-04T19:38:18.127387Z","iopub.execute_input":"2024-06-04T19:38:18.127722Z","iopub.status.idle":"2024-06-04T19:38:18.419065Z","shell.execute_reply.started":"2024-06-04T19:38:18.127697Z","shell.execute_reply":"2024-06-04T19:38:18.418215Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"markdown","source":"# Training Algorithms on the MNIST Dataset","metadata":{}},{"cell_type":"code","source":"from sklearn.datasets import make_classification\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVC\nfrom sklearn.linear_model import LogisticRegression\nfrom xgboost import XGBClassifier\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\n\n# Define classifiers\nclassifiers = {\n    'RandomForestClassifier': RandomForestClassifier(),\n    'KNeighborsClassifier': KNeighborsClassifier(),\n    'SupportVectorMachine': make_pipeline(StandardScaler(), SVC()),\n    'MultiLayerPerceptron': make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000)),\n    'LogisticRegression': LogisticRegression(),\n    'XgboostClassifier': XGBClassifier()\n}\n\n# Train and evaluate each classifier\naccuracies = {}\n\nfor name, clf in classifiers.items():\n    # Train the classifier\n    clf.fit(X_train, y_train)\n    # Predict on the test set\n    y_pred = clf.predict(X_test)\n    # Calculate accuracy\n    accuracy = accuracy_score(y_test, y_pred)\n    # Store accuracy\n    accuracies[name] = accuracy\n    print(f\"{name}: {accuracy:.4f}\")","metadata":{"execution":{"iopub.status.busy":"2024-06-04T19:41:54.58206Z","iopub.execute_input":"2024-06-04T19:41:54.582925Z","iopub.status.idle":"2024-06-04T19:56:39.178092Z","shell.execute_reply.started":"2024-06-04T19:41:54.582893Z","shell.execute_reply":"2024-06-04T19:56:39.177356Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"import tensorflow as tf\nfrom tensorflow.keras import layers, models\nfrom tensorflow.keras.utils import to_categorical\n\n\n# Normalize the input images to the range [0, 1]\nX_train = X_train.astype('float32') / 255.0\nX_test = X_test.astype('float32') / 255.0\n\n# Reshape the data to include the channel dimension (required for CNNs)\nX_train = X_train.reshape((X_train.shape[0], 28, 28, 1))\nX_test = X_test.reshape((X_test.shape[0], 28, 28, 1))\ny_train = to_categorical(y_train, 10)\ny_test = to_categorical(y_test, 10)\n\n# Define the CNN model\nmodel = models.Sequential()\n\n# First convolutional layer\nmodel.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))\nmodel.add(layers.MaxPooling2D((2, 2)))\n\n# Second convolutional layer\nmodel.add(layers.Conv2D(256, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\n\n# Third convolutional layer\nmodel.add(layers.Conv2D(132, (3, 3), activation='relu'))\nmodel.add(layers.MaxPooling2D((2, 2)))\n\n# Fourth convolutional layer\n# Fourth convolutional layer with padding\nmodel.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))\n\n\n# Flatten the output from the convolutional layers\nmodel.add(layers.Flatten())\n\n# Fully connected layer\nmodel.add(layers.Dense(64, activation='relu'))\n\n# Output layer with softmax activation for classification\nmodel.add(layers.Dense(10, activation='softmax'))\n\n# Compile the model\nmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n\n# Print the model summary\nmodel.summary()\n\n# Train the model\nhistory = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)\n\n# Evaluate the model on the test set\ntest_loss, test_acc = model.evaluate(X_test, y_test)\nprint(f'Test accuracy: {test_acc:.4f}')\n","metadata":{"execution":{"iopub.status.busy":"2024-06-04T12:21:04.096215Z","iopub.execute_input":"2024-06-04T12:21:04.096955Z","iopub.status.idle":"2024-06-04T12:23:47.430024Z","shell.execute_reply.started":"2024-06-04T12:21:04.096923Z","shell.execute_reply":"2024-06-04T12:23:47.42905Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"from tabulate import tabulate\n\ndef create_table(accuracies):\n\n\n  # Multiply all values by 100\n  accuracies = {name: accuracy * 100 for name, accuracy in accuracies.items()}\n\n  # Add CNN and test_acc\n  accuracies[\"CNN\"] = test_acc * 100\n\n  # Create the table\n  table = tabulate(accuracies.items(), headers=[\"Model\", \"Accuracy (%)\"], tablefmt=\"grid\")\n\n  return table\n\n\ntable = create_table(accuracies)\n\nprint(table)","metadata":{"execution":{"iopub.status.busy":"2024-06-04T10:14:01.013354Z","iopub.execute_input":"2024-06-04T10:14:01.014082Z","iopub.status.idle":"2024-06-04T10:14:01.021899Z","shell.execute_reply.started":"2024-06-04T10:14:01.014049Z","shell.execute_reply":"2024-06-04T10:14:01.020942Z"},"trusted":true},"execution_count":null,"outputs":[]}]}