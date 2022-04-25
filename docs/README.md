
# Tensorflow Vs. Flax

Author(s): Bradly Whitman, Deepika Venkatesan, Ryan Vennemeyer

## Abstract: Briefly describe your problem, approach, and key results. Should be no more than 300 words.


When comparing two strings there is more to know then just the words being read but the meaning behind them. For model generation we use Adam, a method that is able to store both the individual learning rate of the RMSProp and the weighted average of momentum. Momentum and RMSProp are methods used to take in formation in and train machines and adam is the combination of the two methods. This is how we are able to find out the difference between them are. When processing the dataset it is more efficient to use the JAX import so that you are processing in a more efficient way to save time and processing power. First the NPL is tested without the JAX library to know how long and inefficient the process is. Subsequently, when the code is run with the JAX framework the difference in runtime can be observed. Time is essential when it comes to processing data and if things are not optimized, bigger datasets will not be usable due to inefficiency, lowering the value of the machine. 


## Introduction (10%): Describe the problem you are working on, why it’s important, and an overview of your results


If you have two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related. Your task is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses. To make things more interesting, the train and test set include text in fifteen different languages! You can find more details on the dataset by reviewing the Data page.


NLP is important because it helps resolve ambiguity in language and adds useful numeric structure to the data for many downstream applications, such as speech recognition or text analytics. The reason that this project is important is because of the relationships between sentences. The application of this could be impactful for fact-checking, identifying fake news, analyzing text, and much more.


The resulting efficiency was much higher with the use of flax and the incorporation of JAX into the original model. From an average 228 to 115 seconds per iteration, there was nearly a 99% increase in efficiency overall. The use of a larger data set–double the size of the original–did not affect the overall accuracy of predictions.



## Related Work (10%): Discuss published work that relates to your project. How is your approach similar or different from others? 

https://www.kaggle.com/code/anasofiauzsoy/tutorial-notebook/notebook


The tutorial provided by the Kaggle competition was the base for our first test. We used this source to analyze the sample dataset and record the results that could be obtained without the use of the JAX library. 



https://towardsdatascience.com/5-nlp-models-that-you-need-to-know-about-754594a3225b



This paper explains the existing language processing models that can be used in the development of NLIs, such as BERT, RoBERTa, and XLnet. After recognizing the differences we used BERT to obtain satisfactory results with a less burdensome data set.



https://towardsdatascience.com/creating-word-embeddings-with-jax-c9f144901472



https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008


This source serves as an introduction to RMSProp, momentum, and the overall advantages to using them over traditional SGD. Our model utilizes Adam, a hybridization of both features to reach a happy medium, which has been observed to provide optimal results thus far.



1607.06025.pdf (arxiv.org)



This study by Starc and Mladenic experiments with a new metric for NLU’s. Similar to our project, they utilize the Adam method during model training. This source was used to get a better understanding of NLU model generation and usage.





## Data (10%): Describe the data you are working with for your project. What type of data is it? Where did it come from? How much data are you working with? Did you have to do any preprocessing, filtering, or other special treatment to use this data in your project?


The data we are using are all .csv files acquired from the kaggle API. Test.csv 


The data we used for this project is a dataset provided by the competition tutorial from Kaggle. There are two CSV files, one containing a set to be used for model training, and another for testing. The training set contains 12121 premise-hypothesis pairs with classification (labels), while the test set 5195 pairs does not include their labels. The total size of these files is about 4MB and were used directly in our project. In addition, the premise-hypothesis pairs in these files have fields for ID number, language classification, and the two letter abbreviation of its language



## Methods (30%): Discuss your approach for solving the problems that you set up in the introduction. Why is your approach the right thing to do? Did you consider alternative approaches? You should demonstrate that you have applied ideas and skills built up during the quarter to tackling your problem of choice. It may be helpful to include figures, diagrams, or tables to describe your method or compare it with other methods.


When discussing our problem momentum was brought up because of the way that it runs through data. The way momentum works is that in the beginning of the training it is using a lot of data to get a baseline of what it is learning. When it has a good understanding of what it is learning it will slow down so that it further understands the smaller parts. Another method is called RMSProp that is unpublished and is similar to momentum but is able to use larger learning rates and adaptively adjusts the learning rates for each parameter. With this in mind the combination of the two is called Adam and this works by storing both the individual learning rate of the RMSProp and the  weighted average of momentum. Adam is the method that we decided to use for the Kaggle competition.



## Experiments (30%): Discuss the experiments that you performed to demonstrate that your approach solves the problem. The exact experiments will vary depending on the project, but you might compare with previously published methods, perform an ablation study to determine the impact of various components of your system, experiment with different hyperparameters or architectural choices, use visualization techniques to gain insight into how your model works, discuss common failure modes of your model, etc. You should include graphs, tables, or other figures to illustrate your experimental results.



While TensorFlow is definitely a great tool and provides users with immense support for machine learning in general. Jax works to hone in on maximizing performance above all else. To confirm this we decided to first use tensorflow to run our training data set, and the results were as follows:


Epoch ½


![pic1](https://user-images.githubusercontent.com/51283756/165016421-ac28c06d-cac6-4b03-b65f-043f09fd7b95.jpg)

 
Epoch 2/2


![pic2](https://user-images.githubusercontent.com/51283756/165016468-cc79ec7e-dcba-4c2b-a5c6-7543fdecb141.jpg)


The accuracy was ~.55, with 228-229 iterations / epoch. Now it was time to see how Flax compared to this result. 
For Flax we increased the dataset from 152 to 302 iterations per epoch. And the the iterations / epoch came out to be 229s for the first iteration 


 
And 231 seconds for the second epoch iteration. 


![pic3](https://user-images.githubusercontent.com/51283756/165016571-d79ff1f8-a9e7-4e60-a158-f139ea7167bf.jpg)


While the accuracy did not change from the first iteration, it was already better than the TensorFlow run with double the data set.


The accuracy was ~.47, but the total runtime was lengthy and overall impractical for large scale implementation. 


Model Training Process:


Loss and Accuracy over Epoch 2 - TensorFlow


![pic5](https://user-images.githubusercontent.com/51283756/165016764-6e8a287a-6a0f-4680-98ad-a89d6bf2f0c2.jpg)



![pic6](https://user-images.githubusercontent.com/51283756/165016770-d94e03f8-f5ad-49c4-9308-46cc5c8d4f34.jpg)



## Conclusion (5%) Summarize your key results - what have you learned? Suggest ideas for future extensions or new applications of your ideas.


In conclusion, from the datasets we ran, there is support for the claim that Flax is superior to TensorFlow in both speed/performance, and also accuracy, with a small scale dataset like this. We learned the role TPUs play in speeding up the process of compiling these datasets. We accidentally ran the original TensorFlow compile without enabling the TPU on Google Colab and the runtime was nearly an hour and 30 minutes. This definitely demonstrates the optimization TPUs provide. 
