library(ggplot2)
library(reshape2)
library(ggrepel)
library(R.matlab)
library(plyr)
library(ComplexHeatmap)

rm(list=ls(all=TRUE))

news_source <- "GPHIN"

filein1 <- "~/Projects/covid19_media/gh/results/csv_results/GPHIN_all/Mixmedia_K10/top_20_words.csv"
filein2 <- "~/Projects/covid19_media/gh/results/csv_results/GPHIN_all/Mixmedia_K10/top_20_words_prob.csv"

beta_sel <- read.csv(filein2, row.names=1)

colnames(beta_sel) <- sprintf("M%s", 0:(ncol(beta_sel)-1))

rownames(beta_sel) <- unique(as.character(as.matrix(read.csv(filein1))))

beta_sel <- as.matrix(beta_sel)

hmp <- Heatmap(beta_sel, 
               # column_order = topicOrder, row_order=featOrder, 
               cluster_columns=FALSE, 
               cluster_rows=FALSE,
               name="Prob",
               row_names_side = "left",
               row_dend_side = "right",
               # show_row_dend=FALSE,
               # show_column_dend=FALSE,
               rect_gp = gpar(col = "grey", lwd =0.2),
               row_names_gp = gpar(fontsize=10),
               col=c("white", "red"),
               show_row_names = T,
               row_title="",
               column_title="")


myw <- 6
myh <- 9

hmpout <- sub("csv$", "pdf", filein2)

cairo_pdf(hmpout, width=4.8, height=17)
draw(hmp, heatmap_legend_side = "right", padding=unit(c(5,5,5,5), "mm")) # bottom, left, top, right
dev.off()
system(sprintf("open %s", hmpout))


















