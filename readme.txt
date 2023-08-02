    You can just run the prediction.py or the main_plot.py to plot the figure and see the figure in the directory called
pic.

    The get_features.py is aimed to preprocess the dataset and testset to get the features of them, normalize them and
save the normalized features into a npy file. There are totally ten features. They are  the average distance from S1 to
the next S2, the average distance from S2 to the next S1, the average distance from S1 to the next S1, the average amplitude
of the S1 peak, the average amplitude of the S2 peak, the ratio of the previous two, the average kurtosis and skewness
of S1, and the average kurtosis and skewness of S2.