require(RankAggreg)
require(gtools)
set.seed(21)
dataset_ranking <- c("skindatasetRanking_finalMLRanks","chessdatasetRanking_finalMLRanks","learningcontextRanking_finalMLRanks")
# load the csv dataset file
df_list<- c()
for(n in dataset_ranking){
  df_list <- c(df_list,read.csv(paste("D:/lcif/16032017-IndividualFiles/AnalysisFiles/",n,".csv", sep=""), header=T, sep=',',stringsAsFactors=FALSE))
}


#Rank aggregation using Spearman footrule distance
skin_rank <- t(matrix(unlist(df_list[5]),byrow=T))
chess_rank <- t(matrix(unlist(df_list[10]),byrow=T))
lc_rank <- t(matrix(unlist(df_list[15]),byrow=T))
# syst_rank<- t(matrix(unlist(df["syst_agg_score_rank"]),byrow=T))
parameters <- rbind(lc_rank,skin_rank,chess_rank)
wmat1<- matrix(100:100,1,13)
wmat2<- matrix(.5:.5,1,13)
wmat3<-matrix(.5:.5,1,13)
w<- rbind(wmat1,wmat2,wmat3)
(rank <- RankAggreg(parameters,k=13,method="CE",weights=NULL,distance="Kendall", rho=.1,verbose=FALSE))
final_rank_df <- cbind(df_list[4],data.frame(rank$top.list))

write.csv(final_rank_df, file="D:/lcif/16032017-IndividualFiles/AnalysisFiles/FinalRankAfterAggregation.csv",row.names = F)
