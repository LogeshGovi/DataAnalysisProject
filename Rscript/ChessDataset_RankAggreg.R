require(RankAggreg)
require(gtools)

set.seed(21)
# load the csv dataset file
df <- read.csv("D:/lcif/16032017-IndividualFiles/AnalysisFiles/chessdatasetRanking.csv", header=T, sep=',')

# choose the columns that are to be kept for the problem

# keeps <- c("ssize", "nr_trtime_clus","nr_trtime_rand","nr_trtime_strat",
#            "nr_trtime_syst","nr_tetime_clus","nr_tetime_rand","nr_tetime_strat",
#            "nr_tetime_syst","trerr_clus","trerr_rand","trerr_strat","trerr_syst",
#            "teerr_clus","teerr_rand","teerr_strat","teerr_syst","ml_label","clus_agg_score",  
#            "rand_agg_score","strat_agg_score","syst_agg_score")
# df <- df[keeps]

#Create a dataframe to act on specific columns
# drops <- c("ssize","ml_label","nr_trtime_clus","nr_trtime_rand","nr_trtime_strat",
#            "nr_trtime_syst","nr_tetime_clus","nr_tetime_rand","nr_tetime_strat",
#            "nr_tetime_syst","trerr_clus","trerr_rand","trerr_strat","trerr_syst",
#            "teerr_clus","teerr_rand","teerr_strat","teerr_syst")
# 
# 
# df1 <- df[, !(names(df) %in% drops)]

keeps <-c("clus_agg_score","rand_agg_score","strat_agg_score","syst_agg_score")
df1 <- df[keeps]

# Create the rank columns and append them to the dataframe
for(i in names(df1)){
  df[paste(i,"rank",sep="_")]<-with(df,rank(df[i],ties.method="random"))
}

#Rank aggregation using Spearman footrule distance
clus_rank <- t(matrix(unlist(df["clus_agg_score_rank"]),byrow=T))
rand_rank <- t(matrix(unlist(df["rand_agg_score_rank"]),byrow=T))
strat_rank <- t(matrix(unlist(df["strat_agg_score_rank"]),byrow=T))
syst_rank<- t(matrix(unlist(df["syst_agg_score_rank"]),byrow=T))
parameters <- rbind(clus_rank,rand_rank,strat_rank,syst_rank)
wmat1<- matrix(1:1,2,104)
wmat2<- matrix(.1:.1,2,104)
w<- rbind(wmat1,wmat2)
(rank <- RankAggreg(parameters,k=104,method="CE",weights=w,distance="Spearman", rho=.1,verbose=FALSE))

df[,"total_agg_rank"] <- rank[1]


# write the dataframe with ranks to disk
write.csv(df,file="D:/lcif/16032017-IndividualFiles/AnalysisFiles/chessdatasetRanking_withranks.csv",row.names = F)
