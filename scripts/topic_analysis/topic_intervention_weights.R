library(ggplot2)
library(reshape2)
library(ggrepel)
library(R.matlab)
library(plyr)
library(ComplexHeatmap)

rm(list=ls(all=TRUE))

news_source <- "GPHIN"

filein <- "~/Projects/covid19_media/gh/results/csv_results/GPHIN_all/Mixmedia_K10/interventions_over_topics.csv"

omega <- read.csv(filein, row.names=1)

K <- 10

hm <- Heatmap(as.matrix(omega), border="black", 
              rect_gp = gpar(col = "darkgrey", lwd = 1), 
              cluster_columns=FALSE, 
              cluster_rows=TRUE,
              name = "", 
              heatmap_legend_param = list(direction = "horizontal"),
              column_title="")

hmout <- sub("csv$", "pdf", filein)

cairo_pdf(hmout, width=6.4, height=4.8)
draw(hm, heatmap_legend_side = "bottom", padding=unit(c(5,5,5,30), "mm")) # bottom, left, top, right
dev.off()

system(sprintf("open %s", hmout))



























