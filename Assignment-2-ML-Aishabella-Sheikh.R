# Assignment 2 - Machine Learning----------------------------------------------------------

## Scenario: 
# As a data scientist employed by Today's Network, you will have to build a Natural Language 
# Processing (NLP) model to combat fake content problems. It is believed that machine-learning 
# technologies hold promise for significantly automating parts of the procedure human fact-checkers
#  use today to determine if a story is real or a hoax.

# Activate Required Libraries--------------------------------------------------------------

library ('readr')   # to read the data
library ('tidyverse') # data import + tidying
library('tidytext') #for NLP
library ('party') # recursive partitioning
library ('dplyr') #for data manipulation
library ('stringr') # for data manipulation 
library ('Hmisc') # for data analysis
library ('devtools') # simplify development of r packages
library('Amelia') #visualize missing values
library('textdata') #for sentiment analysis
library('ISLR') #for data analysis and manipulation
library('ggplot2') #for data visualization
library('GGally') #data visualization
library('tm') # for text mining
library('SnowballC') # for stemming words
library('corpus') #for text analysis
library('caret') #for classification and regression training
library('rpart') #models
library('rpart.plot') #visualize rpart
library('e1071') #SVM library
library('quanteda') #text analysis
library('irlba') #feature extraction
library('randomForest') # modelling
library('doSNOW') #for clusters
library('ROSE') #for random over sampling
library('caTools') #for train/test split
library('lsa') #latent semantic analysis
library('h2o') #machine learning
library('wordcloud') # word-cloud generator
library('RColorBrewer') #colour palette
library('corrgram') #correlation
library('corrplot') #correlation plot
library('car') #companion to applied regression
library('moments') #skewness, kurtosis tests
library('gridExtra') #extensions to grid system
library ('MLmetrics') #evaluation
library('class') #classification
library ('testthat') #check correct behaviour of code
library ('blorr') #logistic regression
library ('magrittr') #forward pipe operator %>%
library ('ROCR') #visualize performance of scoring classifiers
library ('pROC') #display and analyze ROC curves

# Import Data---------------------------------------------------------------------------------------
news <- read.csv (file.choose(), header = T, sep = ";")

#view head and tail of dataset
head (news)
tail (news)

#view as tibble
as_tibble(news)

#view data as string
str(news)

#view data as spreadsheet
view(news)

#view summary of data
summary(news)

# Explore & Transform Data--------------------------------------------------------------------------

## Check & Visualize missing values-----------------------------------------------------------------

#check for missing values
sum(is.na(news))

#Visualize missing data with Missmap
missmap(news)

## Remove Duplicates--------------------------------------------------------------------------------
#check for duplicates
subset(news,duplicated(news))

news_0 <- news %>% distinct()

#check new dimensions after removal
dim(news_0)

## Clean Labels-------------------------------------------------------------------------------------
#check labels
news_0 %>% group_by(Label) %>% summarise(count=n())

#Remove unlabeled articles
news_1 <- news_0[news_0$Label %in% c('FAKE', 'REAL'),]
news_1 = as.data.frame(news_1)
summary(news_1)

#Convert Label column to factor
news_1$Label <- as.factor(news_1$Label)

## GGplot of FAKE vs REAL Labels Count--------------------------------------------------------------
news_1%>% ggplot(aes(x = Label, fill=Label)) + theme_bw() +
  geom_bar() + labs(title='Barplot of Fake vs Real News') +
  geom_text(aes(label=..count..),
            stat='count', position=position_dodge(0.9), vjust=-0.2)

#view percentages of fake vs real articles
prop.table(table(news_1$Label))

## % Class Distribution Plot -----------------------------------------------------------------------
distplot <- theme (plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data = news_1, aes(x = Label, 
                          y = prop.table(stat(count)), fill = Label,
                          label = scales::percent (prop.table (stat(count))))) +
  geom_bar (position = "dodge") + theme_bw() +
  geom_text (stat = 'count',
             position = position_dodge(.9), 
             vjust = -0.5, 
             size = 3) + 
  scale_x_discrete (labels = c("FAKE", "REAL"))+
  scale_y_continuous (labels = scales::percent)+
  labs (x = 'Class', y = 'Percentage') +
  ggtitle ("Class Distribution %") +
  distplot

## Add Text length column
news_1$Length <- nchar(news_1$Text)
summary(news_1$Length)

as_tibble(news_1)

## Text Length Histogram
ggplot(news_1, aes(x=Length, fill = Label)) +
  theme_bw() + geom_histogram(binwidth = 5) +
  labs(y= "Text Count", x = "Length of Text",
       title = 'Distribution of Text Lengths')

#create subsets to view difference in mean text length
newsfake <- subset(news_1, Label == 'FAKE')
newsreal <- subset(news_1, Label == 'REAL')
print('Text Length for Fake:')
summary(newsfake$Length)
print('Text Length for Real News:')
summary(newsreal$Length)

#check outliers according to Length
subset(news_1, Length>500)

#remove outliers
news_2 <- subset(news_1, !(Length>500))
dim(news_2)

## Histogram of Text Length without outliers--------------------------------------------------------
ggplot(news_2, aes(x=Length, fill = Label)) +
  theme_bw() + geom_histogram(binwidth = 5) +
  labs(y= "Text Count", x = "Length of Text",
       title = 'Distribution of Text Lengths with Labels')

# compare updated text length stats for fake and real
newsfake <- subset(news_2, Label == 'FAKE')
newsreal <- subset(news_2, Label == 'REAL')
 
summary(newsfake$Length)
summary(newsreal$Length)

## Word Count per Text

#code below sourced from: 
#https://stackoverflow.com/questions/31398077/counting-the-total-number-of-words-in-of-rows-of-a-dataframe
news_2$Words <- sapply(news_2$Text, function(x) 
  length(unlist(strsplit(as.character(x), "\\W+"))))
as_tibble(news_2)

## Boxplot of Word Count-------------------------------------------------------
ggplot (news_2, aes (x = factor(Label), y = Words, fill=Label)) + geom_boxplot() + 
  labs (x = 'Label', y = 'Number of Words') + theme_bw() +
  ggtitle ("Distribution of Word Count by Label")

#check stats of word count for each label
newsfake <- subset(news_2, Label == 'FAKE')
newsreal <- subset(news_2, Label == 'REAL')
summary(newsfake$Words)
summary(newsreal$Words)

## Subset and Shuffle data----------------------------------------------------------------------------
set.seed (2022) 
news_3 <- news_2[order(runif(n=5000)),]

#convert to data frame
as.data.frame(news_3)
#reset index
rownames(news_3) <- NULL
#view
as_tibble(news_3)
head(news_3)
tail(news_3)

# Natural Language Processing (NLP)------------------------------------

## Data Pre-Processing ------------------------------------------------------------------------------

## Create Corpus of Text column 
corpus <- Corpus (VectorSource (news_3$Text))

### Change all words to lowercase
news.corpus <- tm_map (corpus, content_transformer (tolower)) 
### Remove Stop Words
news.corpus <- tm_map (news.corpus, removeWords, stopwords("english"))
### Remove Punctuation
news.corpus <- tm_map (news.corpus, removePunctuation)
### Remove Numbers
news.corpus <- tm_map (news.corpus, removeNumbers)
### Strip white space
news.corpus <- tm_map (news.corpus, stripWhitespace)

### Stem words using SnowballC library
news.corpus <- tm_map (news.corpus, stemDocument)

#compare
news_3$Text[1:5]
inspect (news.corpus [1:5])

### Create Document Term Matrix ------------------------------------------
news.corpus.dtm <- DocumentTermMatrix (news.corpus)
news.corpus.dtm = removeSparseTerms(news.corpus.dtm, 0.999) 
inspect(news.corpus.dtm)

### Split Data into Train and Test Sets----------------------------------
#split raw data 
n <- nrow (news_3)
raw.news.train <- news_3 [1:round(.7 * n),]
raw.news.test  <- news_3 [(round(.7 * n)+1):n,]

#split corpus 
nn <- length (news.corpus)
news.corpus.train <- news.corpus [1:round(.7 * nn)]
news.corpus.test  <- news.corpus [(round(.7 * nn)+1):nn]

#split dtm 
nnn <- nrow (news.corpus.dtm)
news.corpus.dtm.train <- news.corpus.dtm[1:round(.7 * nnn),]
news.corpus.dtm.test  <- news.corpus.dtm[(round(.7 * nnn)+1):nnn,]

### Create Data frame of DTM matrix-------------------------------------
newsdf <- as.data.frame(as.matrix(news.corpus.dtm.train))

#append labels to dtm
newsdf <- cbind(Label=raw.news.train$Label, newsdf)

#make column names syntactically valid
colnames(newsdf) <- make.names(colnames(newsdf))
as_tibble(newsdf)
dim(newsdf)

### Bag of Words for Fake and Real subsets-----------------------------------------
## Split into Fake and Real subsets
fake <- subset (raw.news.train, Label == 'FAKE', select = c(Text, Label))
real <- subset (raw.news.train, Label == 'REAL', select = c(Text, Label))

## Create corpus for fake and real subsets 
corpus_fake <- Corpus(VectorSource(fake$Text))
corpus_real <- Corpus(VectorSource(real$Text))

clean.corpus_fake <- tm_map(corpus_fake, content_transformer(tolower))
clean.corpus_real <- tm_map(corpus_real, content_transformer(tolower))

clean.corpus_fake <- tm_map(clean.corpus_fake, removeNumbers)
clean.corpus_real <- tm_map(clean.corpus_real, removeNumbers)

clean.corpus_fake <- tm_map(clean.corpus_fake, removeWords, stopwords("english"))
clean.corpus_real <- tm_map(clean.corpus_real, removeWords, stopwords("english"))

clean.corpus_fake <- tm_map(clean.corpus_fake, removePunctuation)
clean.corpus_real <- tm_map(clean.corpus_real, removePunctuation)

clean.corpus_fake <- tm_map(clean.corpus_fake, stripWhitespace)
clean.corpus_real <- tm_map(clean.corpus_real, stripWhitespace)

clean.corpus_fake <- tm_map (clean.corpus_fake, stemDocument)
clean.corpus_real <- tm_map (clean.corpus_real, stemDocument)

inspect(clean.corpus_fake [1:5])
inspect(clean.corpus_real [1:5])

#create document term matrix of fake terms
dtm.fake <- DocumentTermMatrix(clean.corpus_fake)
dtm.fake <- removeSparseTerms(dtm.fake, 0.999)
inspect(dtm.fake)

#create document term matrix of real terms
dtm.real <- DocumentTermMatrix(clean.corpus_real)
dtm.real <- removeSparseTerms(dtm.real, 0.999)
inspect(dtm.real)

## Word Cloud------------------------------------------------------------
wordcloud(news.corpus.train, min.freq = 50, random.order = FALSE, 
          colors=brewer.pal (8, "Dark2"), scale=c(6,.5))

### Word Cloud of Fake Terms-------------------------------------------------------
wordcloud (clean.corpus_fake, max.words = 50, 
           random.order = FALSE, colors = brewer.pal(8, "Spectral")[1:3],
           scale=c(6,.5))

### Word Cloud of Real Terms----------------------------------------------------
wordcloud(clean.corpus_real, max.words = 50, scale=c(6,.5),
           random.order = FALSE, colors = rev(brewer.pal(8, "Spectral")[6:8]))

## Frequency Barplots-------------------------------------------
### Fake Terms Barplot
fakefreq <- colSums(as.matrix(dtm.fake))
fakefreq <- sort(fakefreq, decreasing = T)
barplot(fakefreq[1:15], col = 'red', las=1, xlab = 'Frequency', 
        xlim=c(0,700), main='Most Frequent Terms in Fake News', horiz=T)

### Real Terms Barplot
realfreq <- colSums(as.matrix(dtm.real))
realfreq <- sort(realfreq, decreasing = T)
barplot(realfreq[1:15], col = 'green', las=1, xlab = 'Frequency', 
        xlim=c(0,300), main='Most Frequent Terms in Real News', horiz=T)

## Bigrams ------------------------------------------------------
# code source: https://www.r-bloggers.com/2019/08/how-to-create-unigrams-bigrams-and-n-grams-of-app-reviews/
# Create fake bigrams dataframe
fakebigrams <- as.data.frame(fake %>% 
  unnest_tokens(word, Text, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE))

as_tibble(fakebigrams)

#Create real bigrams dataframe
realbigrams <- as.data.frame(real %>% 
  unnest_tokens(word, Text, token = "ngrams", n = 2) %>% 
  separate(word, c("word1", "word2"), sep = " ") %>% 
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>% 
  unite(word,word1, word2, sep = " ") %>% 
  count(word, sort = TRUE))

as_tibble(realbigrams)

### Bigram Word Clouds--------------------------------------------
# Fake bigrams wordcloud
wordcloud (fakebigrams$word, fakebigrams$n, max.words=30,
           random.order = F, colors =brewer.pal(8, "Spectral")[1:3])

# Real bigrams wordcloud
wordcloud (realbigrams$word, realbigrams$n, max.words=40, scale = c(3,.5),
           random.order = F, colors =brewer.pal(8, "BrBG")[6:8])

### Bigram Frequency Plots--------------------------------------
# Frequent Bigrams in Fake News
ggplot(head(fakebigrams, 15), aes(reorder(word,n), n)) +
  geom_bar(stat = "identity", fill = "orange") + coord_flip() +
  xlab("Bigrams") + ylab("Frequency") + ylim(0,115) +
  ggtitle("Most Frequent Bigrams in Fake News")

# Frequent Bigrams in Real News
ggplot(head(realbigrams, 15), aes(reorder(word,n), n)) +
  geom_bar(stat = "identity", fill = "cyan") + coord_flip() +
  xlab("Bigrams") + ylab("Frequency") + ylim(0,35) +
  ggtitle("Most Frequent Bigrams in Real News")

### Word Associations--------------------
# Associations between common fake terms
findAssocs (dtm.fake, c("say","state", 'obama', 'health', 'percent', 'vote', 'tax', 'one'),
            corlimit=0.1)
# Associations between common real terms
findAssocs (dtm.real, c("say","state", 'obama', 'health', 'percent', 'vote', 'time', 'tax'),
            corlimit=0.1)

## Sentiment Analysis-------------------------------------
#codes sourced from: https://stackoverflow.com/questions/50200788/sentiment-analysis-afinn-in-r
# Fake sentiments
#add index column
fake$index <- 1:nrow(fake)
#calculate score for each token in text
fake_sentiment <- fake %>%
  unnest_tokens(word, Text, token = "words") %>%
  inner_join(get_sentiments("afinn"))

#add scores together
fake_sentiments <- fake %>% left_join(fake_sentiment %>%
              group_by(index) %>%
              summarise(score = sum(value))) %>%
  replace_na(list(score = 0))

#remove index column
fake_sentiments <- fake_sentiments[-3]
as_tibble(fake_sentiments)
summary(fake_sentiments)

## Histogram of Fake news affinities------------------------------------------------- 
ggplot(fake_sentiments, aes(x=score)) +
  theme_bw() + geom_histogram(binwidth = 0.5, fill='red') +
  labs(y= "Text Count", x = "Affinity",
       title = 'Distribution of Sentiment Affinity of Fake News')

# Real Sentiments
#add index column
real$index <- 1:nrow(real)
#calculate score for each token in text
real_sentiment <- real %>%
  unnest_tokens(word, Text, token = "words") %>%
  inner_join(get_sentiments("afinn"))

#add scores together and join with data
real_sentiments <- real %>% left_join(real_sentiment %>%group_by(index) %>%
                  summarise(score = sum(value))) %>% replace_na(list(score = 0))
#remove index column
real_sentiments <- real_sentiments[-3]
as_tibble(real_sentiments)
summary(real_sentiments)

## Histogram of Real news affinities-----------------------------------------------
ggplot(real_sentiments, aes(x=score)) +
  theme_bw() + geom_histogram(binwidth = 0.5, fill='turquoise') +
  labs(y= "Text Count", x = "Affinity",
       title = 'Distribution of Sentiment Affinity of Real News')

## combine both fake and real sentiments
all_sent <- rbind(fake_sentiments, real_sentiments) #combine fake and real

## Histogram of all sentiment affinities-------------------------------------
ggplot(all_sent, aes(x=score, fill=Label)) +
  theme_bw() + geom_histogram(binwidth = 1, position='fill') +
  labs(y= "Proportion of Texts", x = "Affinity",
       title = 'Distribution of Sentiment Affinity')

# Cross Validation 1 Decision Tree------------------------------------------
#create stratified folds
set.seed(100)
cv.folds <- createMultiFolds(newsdf$Label, k = 5, times=3)

#set value of K parameter
cv.cntrl <- trainControl(method = "repeatedcv", number = 5,
                         repeats = 3, index = cv.folds) 

#create cluster using doSNOW to work on 2 logical cores
cl <- makeCluster(2, type = 'SOCK')
registerDoSNOW(cl)

#create a single decision tree
rpart.cv.1 <- train(Label~., data = newsdf, method = 'rpart',
                    trControl = cv.cntrl, tuneLength = 5)

# when pre-processing is done, stop cluster
stopCluster(cl)

#results
rpart.cv.1

# Create TF-IDF-------------------------------------------------------------
tfidf <- weightTfIdf(news.corpus.dtm.train)
news.tfidf <- as.data.frame(as.matrix(tfidf))

#append labels
news.tfidf <- cbind(Label=newsdf$Label, news.tfidf)
#make column names syntactically valid
colnames(news.tfidf) <- make.names(colnames(news.tfidf))
as_tibble(news.tfidf)

#Fake Key Words
faketfidf <- subset(news.tfidf, Label == 'FAKE')
#exclude label column 
fakefreq.tfidf <- colSums(faketfidf[,-1])
#sort by frequency
fakefreq.tfidf <- sort(fakefreq.tfidf, decreasing = T)
as_tibble(fakefreq.tfidf)

# TF-IDF Weighted Fake Terms Plot
barplot(fakefreq.tfidf[1:15], col = 'orange', las=1, 
        main='Top 15 Terms in Fake News based on TF-IDF Weight', 
        horiz=T, cex.names=0.8, xlim = c(0,160), xlab = 'Weight (Sum)')

# Real Key Words
realtfidf <- subset(news.tfidf, Label == 'REAL')
realfreq.tfidf <- colSums(realtfidf[,-1]) 
realfreq.tfidf <- sort(realfreq.tfidf, decreasing = T)
as_tibble(realfreq.tfidf)

# TF-IDF Weighted Fake Terms Plot
barplot(realfreq.tfidf[1:15], col = 'turquoise', las=1, 
        main='Top 15 Terms in Real News based on TF-IDF Weight', 
        horiz=T, cex.names=0.8, xlim = c(0,85), xlab = 'Weight (Sum)')

## Cross Validation 2 Decision Tree----------------------------------------
set.seed(100)

#create cluster using doSNOW to work on 2 logical cores
cl <- makeCluster(2, type = 'SOCK')
registerDoSNOW(cl)

#create a single decision tree for tf-idf
rpart.cv.2 <- train(Label~., data = news.tfidf, method = 'rpart',
                    trControl = cv.cntrl, tuneLength = 5)

# when pre-processing is done, stop cluster
stopCluster(cl)

#results
rpart.cv.2

# Singular Value Decomposition----------------------------------------------
#The 30 most relevant features need to be extracted to perform latent semantic analysis (LSA) 
#(source: https://online.datasciencedojo.com/course/text-analytics-with-r/model-building-and-evaluation/svd-with-r)
tfidf_m <- as.matrix(news.tfidf[-1]) #convert to matrix
as_tibble(tfidf_m)
#use library irlba to perform SVD for LSA
train.irlba <- irlba(t(tfidf_m), nv = 30, maxit = 300)
summary(train.irlba$v) 

#project new data into SVD semantic space
sigma.inverse <- 1/train.irlba$d #d maps to sigma
u.transpose <- t(train.irlba$u) #take u matrix and transpose 
document <- tfidf_m[1,] #take first document of tfidf
document.hat <- sigma.inverse * u.transpose %*% document #multiply 3 above together

#look at first 15 components of projected document and corresponding row in V matrix
document.hat[1:15]

#Compare with train.irlba$v document
train.irlba$v[1, 1:15]
#The document hat and v matrix values are identical

# create new feature data frame using document semantic space of 30 features
news.svd <- data.frame(Label = news.tfidf$Label, train.irlba$v)
as_tibble(news.svd)

## Cross Validation 3 Decision Tree-----------
set.seed(100)

#create cluster using doSNOW to work on 2 logical cores
cl <- makeCluster(2, type = 'SOCK')
registerDoSNOW(cl)

#create decision tree
rpart.cv.3 <- train(Label~., data = news.svd, method = 'rpart',
                    trControl = cv.cntrl, tuneLength = 5)

# when pre-processing is done, stop cluster
stopCluster(cl)

#Results
rpart.cv.3

# Cosine Similarity------------------------------------
# source: https://online.datasciencedojo.com/course/text-analytics-with-r/model-building-and-evaluation/cosine-similarity
#create matrix to compare similarities
cos.news <- cosine(t(as.matrix(news.svd[,-1])))
dim(cos.news)

#get indexes from training set of fake news
fake.indexes <- which(raw.news.train$Label == 'FAKE')
#create similarity feature
news.svd$FakeSim <- rep(0.0, nrow(raw.news.train))
for (i in 1:nrow(news.svd)) {
  news.svd$FakeSim[i] <- mean(cos.news[i, fake.indexes])
}

#visualize feature with histogram
ggplot(news.svd, aes(x=FakeSim, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 0.001) +
  labs(y='News Count',
       x='Mean Fake News Cosine Similarity',
       title = 'Distribution of Fake vs Real using Fake Cosine Similarity')

#compare fake vs real cosine similarity
summary(news.svd$FakeSim[news.svd$Label == 'FAKE'])
summary(news.svd$FakeSim[news.svd$Label == 'REAL'])
#the distributions are too similar, so this feature will not be included
news.svd <- news.svd[,-32]
dim(news.svd)

# Random Forest & ROSE for Class Imbalance-----------------------------------------------------
#set levels for labels
levels(news.svd$Label) <- c("FAKE", "REAL")
#scale numeric variables
news.svd[,-1] <- scale(news.svd[,-1]) 

## Split data  using caTools
set.seed(100)
newssplit <- sample.split(news.svd$Label, SplitRatio = 0.75)
newstrain <-  subset (news.svd, newssplit == TRUE)
newstest <- subset (news.svd, newssplit == FALSE)

# class ratio initially
table (newstrain$Label)
table(newstest$Label)

# Random Forest for class imbalance
# source: https://www.r-bloggers.com/2021/05/class-imbalance-handling-imbalanced-data-in-r/
set.seed(500)
rftrain <- randomForest(Label~., data = newstrain, importance=TRUE)
rftrain

#Confusion Matrix
set.seed(100)
rfpred <- predict(rftrain, newstest)
confusionMatrix(rfpred, newstest$Label, positive = 'REAL')

## Over Sampling
set.seed(3338)
over <- ovun.sample(Label~., data=newstrain, method = "over", N = 3338)$data
table(over$Label)

### Random Forest Model Over Sampling---------------------------------- 
set.seed(3230)
rfover <- randomForest(Label~., data = over)
rfover
rfoverpred <- predict(rfover, newstest)
confusionMatrix(rfoverpred, newstest$Label, positive = 'REAL')

## Under Sampling 
set.seed(1912)
under <- ovun.sample(Label~., data=newstrain, method = "under", N =1912)$data
table(under$Label)

### Random Forest Model Under Sampling-------------------------------
set.seed(956)
rfunder <- randomForest(Label~., data=under)
rfunder

set.seed(1912)
rfunderpred <- predict(rfunder, newstest)
confusionMatrix(rfunderpred, newstest$Label, positive = 'REAL')

## Both Over and Under Sampling--------------------------------------
both <- ovun.sample(Label~., data=newstrain, method = "both",
                    p = 0.5,
                    seed = 100,
                    N = 2625)$data
table(both$Label)
### Random Forest Model for Both
set.seed(100)
rfboth <-randomForest(Label~., data=both)
rfbothpred <- predict(rfboth, newstest)
confusionMatrix(rfbothpred, newstest$Label, positive = 'REAL')
# Random Forest Model

## Variable Importance Plot------------------
importance(rftrain)
varImpPlot(rftrain, scale=TRUE, cex = 0.7, 
           main = 'Random Forest Variable Importance Plot')

varImpPlot(rfover, scale=TRUE, cex = 0.7)
varImpPlot(rfunder, scale=TRUE, cex = 0.7)
varImpPlot(rfboth, scale=TRUE, cex = 0.7)
## Find Best mtry--------------------------------------------
#code source: https://www.listendata.com/2014/11/random-forest-with-r.html#id-56fce3
set.seed(100)
mtry <- tuneRF(newstrain[-1], newstrain$Label, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
#find best mtry
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

#Build model again with best mtry value
rf <- randomForest(Label~., data = newstrain, mtry=best.m, importance=TRUE, ntree=500)
rf

rfpred = predict(rf, type = "prob")
perf = prediction(rfpred[,2], newstrain$Label)

# Area under curve
auc = performance(perf, "auc")

# True Positive and Negative Rate
tpperf = performance(perf, "tpr","fpr")

# Plot the ROC curve
plot(tpperf, main="ROC Curve for Random Forest (mtry=5)", col=2,lwd=2)
abline(a=0, b=1, lwd=2, lty=2, col="gray")

# Decision Tree -------------------------------------------------------------------------
#create decision tree using rpart
set.seed(100)
dt <- rpart(Label~., data = newstrain, method = 'class')

#plot tree using rpart.plot
rpart.plot(dt, extra = 106, box.palette=c('red', 'green'))

#Evaluate Decision Tree
predict_class <- predict(dt, newstest, type = 'class')
confusionMatrix(data=predict_class, reference=newstest$Label, 
                positive='REAL')

printcp(dt)
print(dt)
plotcp(dt)
summary(dt)


##Tune hyperparameters---------------------------------------------
control <- rpart.control(minsplit = 5,
                         maxdepth = 4,
                         cp=0.012)
set.seed(100)
tune_fit <- rpart(Label~., data = newstrain, method = 'class', 
                  control = control)


accuracy_tune <- function(dt) {
  predict_dt <- predict(dt, newstest, type = 'class')
  tabledt <- table(newstest$Label, predict_dt)
  accuracy_Test <- sum(diag(tabledt)) / sum(tabledt)
  accuracy_Test
}
accuracy_tune(tune_fit)

# Making Predictions on Test Set---------------------------------------
## Pre-process Test Data-----------------------------------------------
# Create TF-IDF
dim(news.corpus.dtm.test)
test.tfidf <- weightTfIdf(news.corpus.dtm.test)

#convert to dataframe
test.tfidf.df <- as.data.frame(as.matrix(test.tfidf))
as_tibble(test.tfidf.df)

#Append labels
testdf <- cbind(Label=raw.news.test$Label, test.tfidf.df)
#make column names syntactically valid
colnames(testdf) <- make.names(colnames(testdf))
as_tibble(testdf)

#Apply SVD projection
test.svd.raw <- t(sigma.inverse * u.transpose %*% t(as.matrix(test.tfidf)))

#add Label column 
test.svd <- data.frame(Label = raw.news.test$Label, test.svd.raw)
#class levels
levels(test.svd$Label) <- c("FAKE", "REAL")
summary(test.svd)

## Apply Random Forest Model-------------------------------------------
rftestpred <- predict(rf, test.svd)
confusionMatrix(rftestpred, test.svd$Label, positive = 'REAL')

## Apply Decision Tree -----------------------------------------------
dttestpred <- predict(tune_fit, test.svd, type='class')
confusionMatrix(dttestpred, test.svd$Label, positive = 'REAL')

##Bonus: SVM
svm.model <- svm(Label ~ ., data = newstrain, kernel = "radial", cost = 1, gamma = 0.1)
svm.predict <- predict(svm.model, test.svd)
confusionMatrix(svm.predict, test.svd$Label, positive = 'REAL')
