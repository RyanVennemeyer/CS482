



Project Report
The following is a suggested structure for your report, as well as the rubric that we will follow when evaluating reports.

Title, Author(s)

Abstract: Briefly describe your problem, approach, and key results. Should be no more than 300 words.

Introduction (10%): Describe the problem you are working on, why it’s important, and an overview of your results
If you have two sentences, there are three ways they could be related: one could entail the other, one could contradict the other, or they could be unrelated. Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related. Your task is to create an NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses. To make things more interesting, the train and test set include text in fifteen different languages! You can find more details on the dataset by reviewing the Data page.


NLP is important because it helps resolve ambiguity in language and adds useful numeric structure to the data for many downstream applications, such as speech recognition or text analytics. The reason that this project is important is because of the relationships between sentences. The application of this could be impactful for fact-checking, identifying fake news, analyzing text, and much more.



Related Work (10%): Discuss published work that relates to your project. How is your approach similar or different from others? 


https://theaisummer.com/jax/


https://towardsdatascience.com/5-nlp-models-that-you-need-to-know-about-754594a3225b


https://towardsdatascience.com/creating-word-embeddings-with-jax-c9f144901472


https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008


https://www.kaggle.com/code/anasofiauzsoy/tutorial-notebook/notebook




Data (10%): Describe the data you are working with for your project. What type of data is it? Where did it come from? How much data are you working with? Did you have to do any preprocessing, filtering, or other special treatment to use this data in your project?


The data we are using are all .csv files acquired from the kaggle API. Test.csv 


The data we used for this project is a dataset provided by the competition tutorial from Kaggle. There are two CSV files, one containing a set to be used for model training, and another for testing. The training set contains 12121 premise-hypothesis pairs with classification (labels), while the test set 5195 pairs does not include their labels. The total size of these files is about 4MB and were used directly in our project. In addition, the premise-hypothesis pairs in these files have fields for ID number, language classification, and the two letter abbreviation of its language



Methods (30%): Discuss your approach for solving the problems that you set up in the introduction. Why is your approach the right thing to do? Did you consider alternative approaches? You should demonstrate that you have applied ideas and skills built up during the quarter to tackling your problem of choice. It may be helpful to include figures, diagrams, or tables to describe your method or compare it with other methods.


When discussing our problem momentum was brought up because of the way that it runs through data. The way momentum works is that in the beginning of the training it is using a lot of data to get a baseline of what it is learning. When it has a good understanding of what it is learning it will slow down so that it further understands the smaller parts. Another method is called RMSProp that is unpublished and is similar to momentum but is able to use larger learning rates and adaptively adjusts the learning rates for each parameter. With this in mind the combination of the two is called Adam and this works by storing both the individual learning rate of the RMSProp and the  weighted average of momentum. Adam is the method that we decided to use for the Kaggle competition.



Experiments (30%): Discuss the experiments that you performed to demonstrate that your approach solves the problem. The exact experiments will vary depending on the project, but you might compare with previously published methods, perform an ablation study to determine the impact of various components of your system, experiment with different hyperparameters or architectural choices, use visualization techniques to gain insight into how your model works, discuss common failure modes of your model, etc. You should include graphs, tables, or other figures to illustrate your experimental results.



While TensorFlow is definitely a great tool and provides users with immense support for machine learning in general. Jax works to hone in on maximizing performance above all else. To confirm this we decided to first use tensorflow to run our training data set, and the results were as follows:
Epoch ½

 
Epoch 2/2


The accuracy was ~.47, but the total runtime was lengthy and overall impractical for large scale implementation. 


Conclusion (5%) Summarize your key results - what have you learned? Suggest ideas for future extensions or new applications of your ideas.



Is your paper clearly written and nicely formatted? Writing / Formatting (5%)


Supplementary Material, not counted toward your 6-8 page limit and submitted as a separate file.


Your supplementary material might include:


Cool videos, interactive visualizations, demos, etc.


Examples of things to not put in your supplementary material:


The entire PyTorch/TensorFlow Github source code.


Any code that is larger than 10 MB.


Model checkpoints.


Submission: You will submit your final report as a markdown + img files under /docs in your Github repo, obviously together with your code and a README file that describes how anyone can run it to replicate your results. It is highly advised to author a Medium article about your work.

