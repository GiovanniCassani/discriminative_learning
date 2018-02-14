library(car)
library(ggplot2)
library(RColorBrewer)
library(DescTools)

setwd("/Users/GCassani/Desktop/Projects/discriminativeLearning/corpus_new_data/gridSearch/")
T = read.table("gridSearch_outcome.csv", header = TRUE, sep = "\t")

pos_tags = c("A", "B", "C", "D", "N", "O", "P", "Q", "V")
new.unambiguous_baseline = c(5445, 3267, 8, 1, 14060, 20, 16, 2, 1635)
names(new.unambiguous_baseline) = pos_tags
new.unambiguous_acc = max(new.unambiguous_baseline) / sum(new.unambiguous_baseline)
new.unambiguous_entr = Entropy(new.unambiguous_baseline, base = length(new.unambiguous_baseline))


str(T)

T$Stress = as.factor(T$Stress)
T$K = as.factor(T$K)
T$F = as.factor(T$F)
T$Vowels = as.factor(T$Vowels)
levels(T$Corpus) <- list(utterances=c("utterances", "corpus_new_data/aggregate_utterances.json"), 
                         words=c("words", "corpus_new_data/aggregate_words.json"))
T$Acc_baseline = new.unambiguous_acc
T$Entr_baseline = new.unambiguous_entr
T$Acc_diff = T$Accuracy - T$Acc_baseline
T$Entr_diff = T$Entropy - T$Entr_baseline

T = within(T, { acc_St.diff = ifelse(T$Acc_diff <= 0, T$Acc_diff / T$Acc_baseline, T$Acc_diff / (1 - T$Acc_baseline))
                entr_St.diff = ifelse(T$Entr_diff <= 0, T$Entr_diff / T$Entr_baseline, T$Entr_diff / (1 - T$Entr_baseline))})

T = within(T, { acc_St.diff_binary = ifelse(T$acc_St.diff < 0, -1, T$acc_St.diff) })

T.copy = T
str(T.copy)

# x coordinates: corpus, outcomes, method, k
T.copy$Method = recode(T$Method, "'freq'=0; 'sum'=2.25;", as.factor.result=FALSE)
T.copy$K = recode(T$K, "'25'=0; '50'=0.5; '75'=1; '100'=1.5;", as.factor.result=FALSE)
T.copy$Outcomes = recode(T$Outcomes, "'lemmas'=0; 'tokens'=0.25;", as.factor.result=FALSE)
T.copy$Corpus = recode(T$Corpus, "'utterances'=0.1; 'words'=0.2;", as.factor.result=FALSE)


# y coordinates: cue type, evaluation, words to flush
T.copy$Evaluation = recode(T$Evaluation, "'count'=0; 'distr'=1.125;", as.factor.result=FALSE)
T.copy$F = recode(T$F, "'0'=0; '50'=0.25; '100'=0.5; '200'=0.75;", as.factor.result=FALSE)
T.copy$Cues = recode(T$Cues, "'triphones'=0.1; 'syllables'=0.2;", as.factor.result=FALSE)

T$X = rowSums(cbind(T.copy$Corpus, T.copy$Outcomes,T.copy$Method, T.copy$K))
T$Y = rowSums(cbind(T.copy$Cues, T.copy$Evaluation, T.copy$F))


alentejo <- c("#93A7CF", "#D7996D", "#1E222F", "#B85AB7", "#4D648C", "#B5BD83", "#E4CDB0", "#607746", "#EA9B63", 
              "#F0944F", "#FEFE80", "#877568", "#DFC3A5", "#2C2E28", "#233A67",  "#F6CC9F", "#526EAD", "#FAFDFE")
myColors <- alentejo[1:nlevels(T$PoS)]
names(myColors) <- levels(T$PoS)
colScale <- scale_colour_manual(name = "PoS", values = myColors)
x.major_breaks = c(-0.125, 2.155, 4.5)
y.major_breaks = c(-0.125, 1.085, 2.2)

# accuracies
ggplot(T, aes(x=T$X, y=T$Y, color=T$PoS, alpha = T$Entropy)) +
    geom_point(aes(size = T$Accuracy)) +
    colScale + 
    scale_size_continuous(range = c(0.25, 7)) +
    scale_alpha_continuous(range = c(0.1, 1)) +
    theme(panel.grid.major = element_line(colour = "white", size = 5), 
          panel.grid.minor = element_line(colour = "white", size = 0.5),
          axis.title.x=element_blank(),  axis.text.x=element_blank(), axis.ticks.x=element_blank(),
          axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(),
          legend.direction = "horizontal", legend.position = "bottom", legend.box = "horizontal",
          legend.text=element_text(size=12)) +
    scale_y_continuous(breaks = y.major_breaks) +
    scale_x_continuous(breaks = x.major_breaks) +
    scale_shape(guide = guide_legend(title.position = "top")) +
    guides(colour = guide_legend(nrow = 1))


# entropy and entropy standardized differences
myPalette <- colorRampPalette(rev(brewer.pal(11, "PRGn")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-1, 1))
ggplot(T, aes(x=T$X, y=T$Y, alpha = T$Entropy)) + 
    geom_point(aes(size = T$Accuracy, colour = T$entr_St.diff)) +
    scale_size_continuous(range = c(0.25, 7)) +
    scale_alpha_continuous(range = c(0.1, 1)) +
    sc +
    theme(panel.grid.major = element_line(colour = "white", size = 5), 
          panel.grid.minor = element_line(colour = "white", size = 0.5),
          axis.title.x=element_blank(),  axis.text.x=element_blank(), axis.ticks.x=element_blank(),
          axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(),
          legend.direction = "horizontal", legend.position = "bottom", legend.box = "horizontal",
          legend.text=element_text(size=12)) +
    scale_y_continuous(breaks = y.major_breaks) +
    scale_x_continuous(breaks = x.major_breaks) +
    scale_shape(guide = guide_legend(title.position = "top")) +
    guides(colour = guide_legend(nrow = 1))




# accuracy ~ entropy
myPalette <- colorRampPalette(rev(brewer.pal(11, "Blues")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-1, 1))
ggplot(T, aes(y=T$Accuracy, x=T$Entropy)) + 
    geom_point(aes(colour = T$acc_St.diff_binary)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    geom_hline(yintercept = new.unambiguous_acc)  +
    geom_vline(xintercept = new.unambiguous_entr) +
    scale_x_continuous(limits = c(0, 1)) + 
    scale_y_continuous(limits = c(0, 1)) +
    ggtitle("New~Unambiguous") +
    ylab("Accuracy") +
    xlab("Entropy") 




# accuracy differences by entropy differences
myPalette <- colorRampPalette(rev(brewer.pal(11, "Blues")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-1, 1))
ggplot(T, aes(x=T$acc_St.diff, y=T$entr_St.diff)) + 
    geom_point(aes(colour = T$Accuracy)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    scale_x_continuous(limits = c(-1, 1)) + 
    scale_y_continuous(limits = c(-1, 1)) +
    ggtitle("New~Unambiguous") +
    ylab("Entropy") +
    xlab("Accuracy") 



# best models: higher accuracy than baseline and entropy within -0.5 and 0.5 standardized entropy difference
best = T[T$acc_St.diff > 0 & T$entr_St.diff > -0.5 & T$entr_St.diff < 0.5,]
best.copy = best
best.copy$K = as.numeric(best.copy$K)
best.copy$F = as.numeric(best.copy$F)

best.freq = best.copy[best.copy$Method == 'freq', ]
best.freq.count = best.copy[best.copy$Method == 'freq' & best.copy$Evaluation == 'count', ]
best.freq.distr = best.copy[best.copy$Method == 'freq' & best.copy$Evaluation == 'distr', ]
myPalette <- colorRampPalette(rev(brewer.pal(11, "PRGn")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-0.5, 0.5))
ggplot(best, aes(x=best$X, y=best$Y)) + 
    geom_point(aes(size = best$acc_St.diff, colour = best$entr_St.diff)) +
    scale_size_continuous(range = c(0.25, 7)) +
    sc +
    theme(panel.grid.major = element_line(colour = "white", size = 5), 
          panel.grid.minor = element_line(colour = "white", size = 0.5),
          axis.title.x=element_blank(),  axis.text.x=element_blank(), axis.ticks.x=element_blank(),
          axis.title.y=element_blank(), axis.text.y=element_blank(), axis.ticks.y=element_blank(),
          legend.direction = "horizontal", legend.position = "bottom", legend.box = "horizontal",
          legend.text=element_text(size=12)) +
    scale_y_continuous(breaks = y.major_breaks) +
    scale_x_continuous(breaks = x.major_breaks) +
    scale_shape(guide = guide_legend(title.position = "top")) +
    guides(colour = guide_legend(nrow = 1))
