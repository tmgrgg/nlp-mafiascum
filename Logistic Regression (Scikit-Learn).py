#!/usr/bin/env python
# coding: utf-8


# Cloning into the repository to obtain files
get_ipython().system('git clone https://bitbucket.org/bopjesvla/thesis.git')
get_ipython().system('cp thesis/src/* .')


# Import the required modules

# Install dependency
get_ipython().system('pip install googledrivedownloader')

# Download the pickle file from the Google Drive link and store it on Colab
gdd.download_file_from_google_drive(file_id = '1-6uBfJVCAUnPua4zmu_wf3eKWEbc97Zu',
                                    dest_path = './thesis/src/24h_of_deception_full_features.pkl',
                                    unzip = False,
                                    showsize = True)

# Read in the data from the pkl file and reset the indices
# in order to get rid of any indices that might be empty
docs = pd.read_pickle('./thesis/src/24h_of_deception_full_features.pkl')
docs = docs.reset_index()
docs


# Instantiate Model


# Instantiate a model and specify the number of splits of the
# dataset that we will be using. In this case, we will be doing 
# a 20-fold Cross Validation on the dataset. Since
model = LogisticRegression(class_weight = {0: 0.25, 1: 0.75}, max_iter = 300)
kf = KFold(n_splits = 20)


# 1. Hand picked features + FastText Wiki
msg24h = docs['msg24h'].values
tpp_ratio = docs['tpp_ratio'].values
spp_ratio = docs['spp_ratio'].values
sentencelength = docs['sentence_length'].values
vectorFastTextwiki = docs['vector_FastText_fasttext-wiki-news-subwords-300'].values
vectorGloVewiki = docs['vector_GloVe_glove-wiki-gigaword-200'].values

# Input
input_combined = np.hstack([sentencelength.reshape(-1, 1), msg24h.reshape(-1, 1), tpp_ratio.reshape(-1, 1), 
                            spp_ratio.reshape(-1, 1), np.vstack(vectorFastTextwiki)])

# Labels
scum = docs['scum'].values


# Combining Sentence Length, Messages / 24 Hours, 3rd-Person Pronoun Ratio,
# 2nd-Person Pronoun Ration and FastText

score_final = 0.0
auroc_final = 0.0
ap_final = 0.0

for train_index, test_index in kf.split(input_combined):
    X_train, X_test = input_combined[train_index], input_combined[test_index]
    Y_train, Y_test = scum[train_index], scum[test_index]

    # Train the model on the Training Dataset
    model.fit(X_train, Y_train)

    # Model makes predictions based on input from the Test Dataset
    predictions = model.predict(X_test)

    # Compute the percentage accuracy of the model's predictions
    score = model.score(X_test, Y_test)

    # Compute the AUROC of the model
    auroc = roc_auc_score(Y_test, predictions)

    # Compute the Average Precision of the model
    average_precision = average_precision_score(Y_test, predictions)

    # Stores the above results so as to obtain the mean performance
    # of the model on the total dataset
    score_final += score
    auroc_final += auroc
    ap_final += average_precision

print("Score:", score_final / 20.0)
print("AUROC:", auroc_final / 20.0)
print("Average Precision:", ap_final / 20.0)


# 2. FastText Wiki

score_final = 0.0
auroc_final = 0.0
ap_final = 0.0

for train_index, test_index in kf.split(vectorFastTextwiki):
    X_train, X_test = vectorFastTextwiki[train_index], vectorFastTextwiki[test_index]
    Y_train, Y_test = scum[train_index], scum[test_index]

    # Train the model on the Training Dataset
    model.fit(X_train.tolist(), Y_train.tolist())

    # Model makes predictions based on input from the Test Dataset
    predictions = model.predict(X_test.tolist())

    # Compute the percentage accuracy of the model's predictions
    score = model.score(X_test.tolist(), Y_test.tolist())

    # Compute the AUROC of the model
    auroc = roc_auc_score(Y_test, predictions)

    # Compute the Average Precision of the model
    average_precision = average_precision_score(Y_test, predictions)

    # Stores the above results so as to obtain the mean performance
    # of the model on the total dataset
    score_final += score
    auroc_final += auroc
    ap_final += average_precision

print("Score:", score_final / 20.0)
print("AUROC:", auroc_final / 20.0)
print("Average Precision:", ap_final / 20.0)


# 3. GloVe Wiki

score_final_2 = 0.0
auroc_final_2 = 0.0
ap_final_2 = 0.0

for train_index, test_index in kf.split(vectorGloVewiki):
    X_train, X_test = vectorGloVewiki[train_index], vectorGloVewiki[test_index]
    Y_train, Y_test = scum[train_index], scum[test_index]

    # Train the model on the Training Dataset
    model.fit(X_train.tolist(), Y_train.tolist())

    # Model makes predictions based on input from the Test Dataset
    predictions = model.predict(X_test.tolist())

    # Compute the percentage accuracy of the model's predictions
    score = model.score(X_test.tolist(), Y_test.tolist())

    # Compute the AUROC of the model
    auroc = roc_auc_score(Y_test.tolist(), predictions)

    # Compute the Average Precision of the model
    average_precision = average_precision_score(Y_test.tolist(), predictions)

    # Stores the above results so as to obtain the mean performance
    # of the model on the total dataset
    score_final_2 += score
    auroc_final_2 += auroc
    ap_final_2 += average_precision

print("Score:", score_final_2 / 20.0)
print("AUROC:", auroc_final_2 / 20.0)
print("Average Precision:", ap_final_2 / 20.0)
