# This demo follows the below blogpost about topic modelling using R.
# https://eight2late.wordpress.com/2015/09/29/a-gentle-introduction-to-topic-modeling-using-r/
# Download the textmining.zip file from this link and put it in a folder.

#-------------------------------------------------------------------------------------------
# Set the current working directory
setwd("D:\\LDA tutorials\\Intro_to_topic_modelling\\textmining_bigram")

# load text mining library
library(tm)

#load files into corpus
#get listing of .txt files in directory
filenames <- list.files(getwd(),pattern="*.txt")

#read files into a character vector
files <- lapply(filenames,readLines)

#create corpus from vector
docs <- Corpus(VectorSource(files))

# See the structure of docs
str(docs)
# The corpos docs is a list of 30 elements i.e. 30 documents.
# and each of the 30 list has two lists called "content" nad "meta".
# "content" has actual text of the document.
# "meta" is again a list of 7 elements. This has meta dat about the document like author, language..etc.

#inspect a particular document in corpus
writeLines(as.character(docs[[30]]))

#---------------------------------------------------------------------------------------------
#start preprocessing

#Transform to lower case
docs <-tm_map(docs,content_transformer(tolower))

#remove potentially problematic symbols
toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " ", x))})
docs <- tm_map(docs, toSpace, "–")
docs <- tm_map(docs, toSpace, "’")
docs <- tm_map(docs, toSpace, "‘")
docs <- tm_map(docs, toSpace, "•")
docs <- tm_map(docs, toSpace, "“")
docs <- tm_map(docs, toSpace, "”")

#remove punctuation
docs <- tm_map(docs, removePunctuation)
#Strip digits
docs <- tm_map(docs, removeNumbers)
#remove stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
#remove whitespace
docs <- tm_map(docs, stripWhitespace)

#Good practice to check every now and then
writeLines(as.character(docs[[30]]))

#Stem document
# A stemming algorithm reduces the words "fishing", "fished", and "fisher" to the root word, "fish"
docs <- tm_map(docs,stemDocument)

#fix up 1) differences between us and aussie english 2) general errors
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "organiz", replacement = "organ")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "organis", replacement = "organ")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "andgovern", replacement = "govern")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "inenterpris", replacement = "enterpris")
docs <- tm_map(docs, content_transformer(gsub),
               pattern = "team-", replacement = "team")

#define and eliminate all custom stopwords
myStopwords <- c("can", "say","one","way","use",
                 "also","howev","tell","will",
                 "much","need","take","tend","even",
                 "like","particular","rather","said",
                 "get","well","make","ask","come","end",
                 "first","two","help","often","may",
                 "might","see","someth","thing","point",
                 "post","look","right","now","think","‘ve ",
                 "‘re ","anoth","put","set","new","good",
                 "want","sure","kind","larg","yes,","day","etc",
                 "quit","sinc","attempt","lack","seen","awar",
                 "littl","ever","moreov","though","found","abl",
                 "enough","far","earli","away","achiev","draw",
                 "last","never","brief","bit","entir","brief",
                 "great","lot","t","s","don","isn","paul","didn","are","n","won","let",
                 "doesn","go","know","yes","lou","couldn")
docs <- tm_map(docs, removeWords, myStopwords)

#inspect a document as a check
writeLines(as.character(docs[[30]]))

#---------------------------------------------------------------------------------------------

#Create document-term matrix
# A document-term matrix or term-document matrix is a mathematical matrix that describes the 
# frequency of terms that occur in a collection of documents. 
# https://en.wikipedia.org/wiki/Document-term_matrix

#str(dtm)

BigramTokenizer <-
  function(x)
    unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

#tdm <- TermDocumentMatrix(docs, control = list(tokenize = BigramTokenizer))
dtm2 <- DocumentTermMatrix(docs, control = list(tokenize = BigramTokenizer))

#str(dtm)
#==========================================  

#convert rownames to filenames
rownames(dtm2) <- filenames

#collapse matrix by summing over columns
# It shows the total count of each terms
freq <- colSums(as.matrix(dtm2))

#length should be total number of terms
length(freq)

#create sort order (descending)
ord <- order(freq,decreasing=TRUE)

#List all terms in decreasing order of freq and write to disk
freq[ord]
write.csv(freq[ord],"word_freq.csv")

# The document term matrix (DTM) produced by the above code will be the main input into the LDA 
# algorithm of the next section.
#---------------------------------------------------------------------------------------------

#-----------------------------------Topic modelling using LDA---------------------------------
# We’ll use the topicmodels package written by Bettina Gruen and Kurt Hornik. 
# Specifically, we’ll use the LDA function with the Gibbs sampling option
# https://en.wikipedia.org/wiki/Gibbs_sampling
# https://en.wikipedia.org/wiki/Random_walk

#load topic models library
library(topicmodels)

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

#Number of topics
k <- 5

#Run LDA using Gibbs sampling
# Start the clock!
ptm <- proc.time()
ldaOut <-LDA(dtm2,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, 
                                                 burnin = burnin, iter = iter, thin=thin))
# Stop the clock
proc.time() - ptm


#write out results
#docs to topics
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))

#top 6 terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,6))
write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))

#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))


#Find relative importance of top 2 topics
topic1ToTopic2 <- lapply(1:nrow(dtm2),function(x)
  sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])

#Find relative importance of second and third most important topics
topic2ToTopic3 <- lapply(1:nrow(dtm2),function(x)
  sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])

#write to file
write.csv(topic1ToTopic2,file=paste("LDAGibbs",k,"Topic1ToTopic2.csv"))
write.csv(topic2ToTopic3,file=paste("LDAGibbs",k,"Topic2ToTopic3.csv"))

