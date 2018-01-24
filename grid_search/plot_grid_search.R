library(car)
library(ggplot2)
library(RColorBrewer)
library(entropy)
source("/Users/GCassani/Desktop/Projects/discriminativeLearning/plots/multiplot.R")

setwd("/Users/GCassani/Desktop/Projects/discriminativeLearning/")
T = read.table("corpus/aggregate_summaryTable.txt", header = TRUE, sep = "\t")

pos_tags = c("A", "B", "C", "D", "N", "O", "P", "Q", "V")
known_ambiguous_baseline = c(432, 171, 18, 2, 2064, 22, 39, 0, 1550)
known_unambiguous_baseline = c(591, 91, 8, 0, 3630, 45, 17, 0, 567)
new_ambiguous_baseline = c(1141, 333, 15, 1, 1603, 43, 30, 67, 963)
new_unambiguous_baseline = c(5436, 3263, 8, 1, 13998, 20, 16, 2, 1632)
names(known_ambiguous_baseline) = pos_tags
names(known_unambiguous_baseline) = pos_tags
names(new_ambiguous_baseline) = pos_tags
names(new_unambiguous_baseline) = pos_tags

new.amb_acc = 0.383
new.unamb_acc = 0.575
known.amb_acc = 0.48
known.unamb_acc = 0.73

new.amb_entr = 0.535
new.unamb_entr = 0.398
known.amb_entr = 0.651
known.unamb_entr = 0.509

str(T)

T$Stress = as.factor(T$Stress)
levels(T$Stress) <- list(nostress=c("nostress", "0"), stress=c("stress", "1"))
T$K = as.factor(T$K)
T$Flush = as.factor(T$Flush)
T$Reduced.vowels = as.factor(T$Reduced.vowels)
levels(T$Reduced.vowels) <- list(reduced=c("reduced", "1"), full=c("full", "0"))

T.copy = T

test.set.type = strsplit(as.character(T.copy$Test.set), "_")
ambiguous = vector("list", length(T.copy$Test.set))
known = vector("list", length(T.copy$Test.set))
acc_baselines = vector("list", length(T.copy$Test.set))
entr_baselines = vector("list", length(T.copy$Test.set))
for (i in 1:NROW(test.set.type)) {
    ambiguous[i] = test.set.type[[i]][3]
    known[i] = test.set.type[[i]][2]
    if (test.set.type[[i]][2] == "new") {
        if (test.set.type[[i]][3] == "ambiguous") {
            acc_baselines[i] = new.amb_acc
            entr_baselines[i] = new.amb_entr
        } else {
            acc_baselines[i] = new.unamb_acc
            entr_baselines[i] = new.unamb_entr
        }
        
    } else {
        if (test.set.type[[i]][3] == "ambiguous") {
            acc_baselines[i] = known.amb_acc
            entr_baselines[i] = known.amb_entr
        } else {
            acc_baselines[i] = known.unamb_acc
            entr_baselines[i] = known.unamb_entr
        }
    }
}
T$Ambiguous = as.factor(unlist(ambiguous))
T$Known = as.factor(unlist(known))
T$Acc_baseline = as.numeric(unlist(acc_baselines))
T$Entr_baseline = as.numeric(unlist(entr_baselines))
T$Acc_diff = T$Accuracy - T$Acc_baseline
T$Entr_diff = T$Entropy - T$Entr_baseline

T = within(T, {
    acc_St.diff = ifelse(T$Acc_diff <= 0, T$Acc_diff / T$Acc_baseline, T$Acc_diff / (1 - T$Acc_baseline))
    entr_St.diff = ifelse(T$Entr_diff <= 0, T$Entr_diff / T$Entr_baseline, T$Entr_diff / (1 - T$Entr_baseline))
})

T = within(T, {
    acc_St.diff_binary = ifelse(T$acc_St.diff < 0, -1, T$acc_St.diff)
})

str(T.copy)

# x coordinates: ambiguousness, training regime, method, reduced vowels, k
T.copy$Ambiguous = recode(T$Ambiguous, "'unambiguous'=0; 'ambiguous'=5;", as.factor.result=FALSE)
T.copy$Training = recode(T$Training, "'utterances'=0; 'words'=2.25;", as.factor.result=FALSE)
T.copy$Method = recode(T$Method, "'sum'=0; 'freq'=1.125;", as.factor.result=FALSE)
T.copy$Reduced.vowels = recode(T$Reduced.vowels, "'reduced'=0; 'full'=0.5;", as.factor.result=FALSE)
T.copy$K = recode(T$K, "'20'=0.1; '50'=0.2; '100'=0.3; '200'=0.4;", as.factor.result=FALSE)

# y coordinates: new/known, cue type, evaluation, stress, words to flush
T.copy$Known = recode(T$Known, "'known'=0; 'new'=5;", as.factor.result=FALSE)
T.copy$Cues = recode(T$Cues, "'triphones'=0; 'syllables'=2.25;", as.factor.result=FALSE)
T.copy$Evaluation = recode(T$Evaluation, "'count'=0; 'distr'=1.125;", as.factor.result=FALSE)
T.copy$Stress = recode(T$Stress, "'nostress'=0; 'stress'=0.5;", as.factor.result=FALSE)
T.copy$Flush = recode(T$Flush, "'0'=0.1; '20'=0.2; '50'=0.3; '100'=0.4;", as.factor.result=FALSE)

T$X = rowSums(cbind(T.copy$Ambiguous, T.copy$Reduced.vowels,T.copy$Training, T.copy$Method, T.copy$K))
T$Y = rowSums(cbind(T.copy$Known, T.copy$Cues, T.copy$Evaluation, T.copy$Stress, T.copy$Flush))

# one plot, with ambiguous and known as further x and y coordinates
alentejo <- c("#93A7CF", "#D7996D", "#1E222F", "#B85AB7", "#4D648C", "#B5BD83", "#E4CDB0", "#607746", "#EA9B63", 
              "#F0944F", "#FEFE80", "#877568", "#DFC3A5", "#2C2E28", "#233A67",  "#F6CC9F", "#526EAD", "#FAFDFE")
myColors <- alentejo[1:nlevels(T$Most.frequent)]
names(myColors) <- levels(T$Most.frequent)
colScale <- scale_colour_manual(name = "Most.frequent", values = myColors)
major.breaks = c(-0.125, 4.7, 9.5)
minor.breaks = c(1.0575, 2.18, 3.325, 6.0575, 7.18, 8.325)
pdf("plots/gridSearch_completePlot.pdf", width = 20, height = 30)
ggplot(T, aes(x=T$X, y=T$Y, color=T$Most.frequent, alpha = T$Entropy)) +
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
    scale_y_continuous(breaks = major.breaks, minor_breaks = minor.breaks) +
    scale_x_continuous(breaks = major.breaks, minor_breaks = minor.breaks) +
    scale_shape(guide = guide_legend(title.position = "top")) +
    guides(colour = guide_legend(nrow = 1))
dev.off()

pdf("plots/gridSearch_entr.diffsPlot.pdf", width = 20, height = 30)
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
    scale_y_continuous(breaks = major.breaks, minor_breaks = minor.breaks) +
    scale_x_continuous(breaks = major.breaks, minor_breaks = minor.breaks) +
    scale_shape(guide = guide_legend(title.position = "top")) +
    guides(colour = guide_legend(nrow = 1))
dev.off()



known_amb = T[T$Test.set == 'aggregate_known_ambiguous',]
known_unamb = T[T$Test.set == 'aggregate_known_unambiguous',]
new_amb = T[T$Test.set == 'aggregate_new_ambiguous',]
new_unamb = T[T$Test.set == 'aggregate_new_unambiguous',]

pdf("plots/gridSearch_Acc~Entr.pdf", width = 20, height = 20)
myPalette <- colorRampPalette(rev(brewer.pal(11, "Blues")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-1, 1))

p1 <- ggplot(new_unamb, aes(y=new_unamb$Accuracy, x=new_unamb$Entropy)) + 
    geom_point(aes(colour = new_unamb$St.diff_binary)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    geom_hline(yintercept = new.unamb_acc
               )  +
    scale_x_continuous(limits = c(0, 1)) + 
    scale_y_continuous(limits = c(0, 1)) +
    ggtitle("New~Unambiguous") +
    ylab("Accuracy") +
    xlab("") +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())

p2 <- ggplot(known_unamb, aes(y=known_unamb$Accuracy, x=known_unamb$Entropy)) + 
    geom_point(aes(colour = known_unamb$St.diff_binary)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    geom_hline(yintercept = known.unamb_acc) +
    scale_x_continuous(limits = c(0, 1)) + 
    scale_y_continuous(limits = c(0, 1)) +
    ggtitle("Known~Unambiguous") +
    ylab("Accuracy") +
    xlab("Entropy")

p3 <- ggplot(new_amb, aes(y=new_amb$Accuracy, x=new_amb$Entropy)) + 
    geom_point(aes(colour = new_amb$St.diff_binary)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    geom_hline(yintercept = new.amb_acc
               )  +
    scale_x_continuous(limits = c(0, 1)) + 
    scale_y_continuous(limits = c(0, 1)) +
    ggtitle("New~Ambiguous") +
    ylab("") +
    xlab("") +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())
    
p4 <- ggplot(known_amb, aes(y=known_amb$Accuracy, x=known_amb$Entropy)) + 
    geom_point(aes(colour = known_amb$St.diff_binary)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    geom_hline(yintercept = known.amb_acc
               )  +
    scale_x_continuous(limits = c(0, 1)) + 
    scale_y_continuous(limits = c(0, 1)) +
    ggtitle("Known~Ambiguous") +
    ylab("") +
    xlab("Entropy") +
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())

multiplot(p1, p2, p3, p4, cols=2)
dev.off()

# sum-distr-triphones always majority baseline




pdf("plots/gridSearch_diffAcc~diffEntr.pdf", width = 20, height = 20)
myPalette <- colorRampPalette(rev(brewer.pal(11, "Blues")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-1, 1))

p1 <- ggplot(new_unamb, aes(x=new_unamb$acc_St.diff, y=new_unamb$entr_St.diff)) + 
    geom_point(aes(colour = new_unamb$Accuracy)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    scale_x_continuous(limits = c(-1, 1)) + 
    scale_y_continuous(limits = c(-1, 1)) +
    ggtitle("New~Unambiguous") +
    ylab("Entropy") +
    xlab("") +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())

p2 <- ggplot(known_unamb, aes(x=known_unamb$acc_St.diff, y=known_unamb$entr_St.diff)) + 
    geom_point(aes(colour = known_unamb$Accuracy)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    scale_x_continuous(limits = c(-1, 1)) + 
    scale_y_continuous(limits = c(-1, 1)) +
    ggtitle("Known~Unambiguous") +
    xlab("Accuracy") +
    ylab("Entropy")

p3 <- ggplot(new_amb, aes(x=new_amb$acc_St.diff, y=new_amb$entr_St.diff)) + 
    geom_point(aes(colour = new_amb$Accuracy)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    scale_x_continuous(limits = c(-1, 1)) + 
    scale_y_continuous(limits = c(-1, 1)) +
    ggtitle("New~Ambiguous") +
    ylab("") +
    xlab("") +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())

p4 <- ggplot(known_amb, aes(x=known_amb$acc_St.diff, y=known_amb$entr_St.diff)) + 
    geom_point(aes(colour = known_amb$Accuracy)) +
    sc + guides(colour=FALSE, size=FALSE) + 
    scale_x_continuous(limits = c(-1, 1)) + 
    scale_y_continuous(limits = c(-1, 1)) +
    ggtitle("Known~Ambiguous") +
    ylab("") +
    xlab("Accuracy") +
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())

multiplot(p1, p2, p3, p4, cols=2)
dev.off()

best = T[T$acc_St.diff > 0 & T$entr_St.diff > -0.25 & T$entr_St.diff < 0.25,]
pdf("plots/gridSearch_best.pdf", width = 20, height = 30)
major.breaks = c(-0.125, 4.7, 9.5)
minor.breaks = c(1.0575, 2.18, 3.325, 6.0575, 7.18, 8.325)
myPalette <- colorRampPalette(rev(brewer.pal(11, "PRGn")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-0.25, 0.25))
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
    scale_y_continuous(breaks = major.breaks, minor_breaks = minor.breaks, limits = c(0,9.5)) +
    scale_x_continuous(breaks = major.breaks, minor_breaks = minor.breaks, limits = c(0,9.5)) +
    scale_shape(guide = guide_legend(title.position = "top")) +
    guides(colour = guide_legend(nrow = 1))
dev.off()
