library(ggplot2)
library(reshape2)
library(ggrepel)
library(R.matlab)
library(plyr)

rm(list=ls(all=TRUE))

news_source <- "GPHIN_all"

K <- 10

filein <- "~/Projects/covid19_media/gh/results/csv_results/GPHIN_all/Mixmedia_K10/topics_over_time.csv"

eta_df <- read.csv(filein)

times <- read.delim("~/Projects/covid19_media/gh/data/data_June16/GPHIN_all/min_df_10/times_map.txt", header=F, stringsAsFactors = F)
times[,1] <- do.call(rbind, strsplit(times[,1], " : "))[,2]

eta_df$week_label <- sub("2020-","",times[,1])[eta_df$time+1]

eta_df <- subset(eta_df, source %in% c("china","canada","united states", "italy", "taiwan", "uganda"))

eta_df <- droplevels(eta_df)

# eta_df$topic <- sprintf("M%s", eta_df$topic)

eta_df$source <- toupper(eta_df$source)

gg <- ggplot(eta_df, aes(x=week_label, y=topic_value, shape=topic)) + 
  geom_point() + 
  # geom_text(aes(label=topic),size=3) +
  ylab("Dynamic topic prior") + 
  facet_wrap(~source, nrow=3) + # space = "free_x"
  xlab("Month-week") + theme_bw() + 
  scale_shape_identity(guide="legend", breaks=unique(eta_df$topic)) +
  theme(axis.text.x = element_text(angle=30,hjust=1))

ggout <- sub("csv$", "pdf", filein)

ggsave(ggout, gg, width=8, height=12)

system(sprintf("open %s", ggout))


























