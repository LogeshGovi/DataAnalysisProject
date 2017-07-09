require(RankAggreg)
require(gtools)
set.seed(21)
# load the csv dataset file
df <- read.csv("D:/lcif/16032017-IndividualFiles/AnalysisFiles/skindatasetRanking.csv", header=T, sep=',')

keeps <-c("clus_agg_score","rand_agg_score","strat_agg_score","syst_agg_score")
df1 <- df[keeps]

# Create the rank columns and append them to the dataframe
for(i in names(df1)){
  df[paste(i,"rank",sep="_")]<-with(df,rank(df[i],ties.method="random"))
}

total_agg_score <- (rowSums(df1**2))**(1/2)
total_agg_score_rank <- rank(total_agg_score,ties.method="random")
df[,"total_agg_score"] <- c(total_agg_score)
df[,"total_agg_score_rank"]<-c(total_agg_score_rank)
temp_list <- c()
cen_values <- c(mean,var,sd)
for(n in cen_values){
  buff <- aggregate(total_agg_score_rank~ml_alg,df,n)[2]
  temp_list <- c(temp_list,buff)
}
rank_df <- data.frame(t(matrix(unlist(temp_list),nrow=3,byrow=T)))
colnames(rank_df)<- c("mean","var","stddev")
ml_label <- aggregate(total_agg_score_rank~ml_alg,df,mean)[1]
rank_list <-rank(rank_df["mean"],ties.method="random")
rank_df[,"ml_label"] <- c(ml_label)
rank_df[,"final_rank"] <- c(rank_list)
ordered_rank <- rank_df[order(rank_df$final_rank),]


# write the dataframe with ranks to disk
write.csv(df,file="D:/lcif/16032017-IndividualFiles/AnalysisFiles/skindatasetRanking_withranks.csv",row.names = F)
write.csv(ordered_rank, file="D:/lcif/16032017-IndividualFiles/AnalysisFiles/skindatasetRanking_finalMLRanks.csv",row.names = F)
