require(RankAggreg)
require(gtools)
# load the csv dataset file
df <- read.csv("D:/lcif/16032017-IndividualFiles/AnalysisFiles/learningcontextRanking.csv", header=T, sep=',')

# choose the columns that are to be kept for the problem

keeps <- c("ssize", "nr_trtime_clus","nr_trtime_rand","nr_trtime_strat",
           "nr_trtime_syst","nr_tetime_clus","nr_tetime_rand","nr_tetime_strat",
           "nr_tetime_syst","trerr_clus","trerr_rand","trerr_strat","trerr_syst",      
           "teerr_clus","teerr_rand","teerr_strat","teerr_syst","ml_label"   )
df <- df[keeps]

#Create a dataframe to act on specific columns
drops <- c("ssize","ml_label")
df1 <- df[, !(names(df) %in% drops)]

# Create the rank columns and append them to the dataframe
for(i in names(df1)){
  df[paste(i,"rank",sep="_")]<-with(df,rank(df[i]))
}

# write the dataframe with ranks to disk
write.csv(df,file="D:/lcif/16032017-IndividualFiles/AnalysisFiles/learningcontextRanking_withranks.csv",row.names = F)

#Rank aggregation using Spearman footrule distance 
trtime <- t(matrix(unlist(df["nr_trtime_clus_rank"]),byrow=T))
tetime <- t(matrix(unlist(df["nr_tetime_clus_rank"]),byrow=T))
trerr <- t(matrix(unlist(df["trerr_clus_rank"]),byrow=T))
teerr<- t(matrix(unlist(df["teerr_clus_rank"]),byrow=T))
parameters <- rbind(trtime,tetime,trerr,teerr)
wmat1<- matrix(1:1,2,104)
wmat2<- matrix(10:10,2,104)
w<- rbind(wmat1,wmat2)
(rank<- RankAggreg(parameters,k=104,method="CE",weights=w,distance="Spearman", rho=.1, verbose=FALSE))
BruteAggreg(parameters,k=104,weights=w,distance=c("Spearman"))
