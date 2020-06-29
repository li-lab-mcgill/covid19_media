library(Rtsne)
library(R.matlab)
library(parallel)
library(ggplot2)
library(ggrepel)
library(ComplexHeatmap)

rm(list=ls(all=TRUE))

# filein <- "~/Projects/covid19_media/gh/results/mixmedia_embeddings/GPHIN_all/mixmedia_GPHIN_all_K_5_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_predictLabels_1_useTime_1_useSource_1_rho.mat"
# inpdir <- "~/Projects/covid19_media/gh/results/mixmedia_embeddings/GPHIN_all2"
# filelist <- list.files(inpdir, full.names = TRUE)
# filein <- filelist[1]
# rho <- readMat(filein)$value
# vocab_filein <- "~/Projects/covid19_media/gh/data/data_June16/GPHIN_all/min_df_10/vocab_map.txt"
# rownames(rho) <- as.character(read.table(vocab_filein, header=F)[,3])

inpdir <- "~/Projects/covid19_media/gh/results/csv_results"

filelist <- list.files(inpdir, "word_embeddings.csv", recursive = TRUE, full.names = TRUE)

my_words <- c("covid", "corona", "sars", "lockdown", "border",  "mask", "repatriate")

for(filein in filelist) {
  
  rho <- unique(as.matrix(read.csv(filein, row.name=1, header=F)))
  
  # filein <- "~/Projects/covid19_media/data/trained_word_emb_aylien.txt"
  # rho <- unique(as.matrix(read.table(filein, row.name=1)))
  
  my_words <- my_words[my_words %in% rownames(rho)]
  
  topk <- 10
  
  my_dist <- function(w, x) {
    sqrt(rowSums((as.matrix(x) - matrix(rep(as.matrix(x[w,,drop=F]), each=nrow(x)), nrow=nrow(x)))^2))
  }
  
  options(mc.cores=detectCores())
  
  neighbor_words <- mclapply(my_words, function(w) {
    rownames(rho)[head(order(my_dist(w, rho)),topk)]
  })
  
  names(neighbor_words) <- my_words
  
  print(basename(filein))
  print(neighbor_words)
  
  rho_sel <- rho[unique(c(my_words, unlist(neighbor_words))),]
  
  hmp <- Heatmap(rho_sel,
                 # column_order = topicOrder, row_order=featOrder,
                 cluster_columns=TRUE,
                 cluster_rows=TRUE,
                 name="embedding",
                 row_names_side = "left",
                 row_dend_side = "right",
                 # show_row_dend=FALSE,
                 show_column_dend=FALSE,
                 show_column_names = FALSE,
                 # rect_gp = gpar(col = "grey", lwd =0.2),
                 row_names_gp = gpar(fontsize=10),
                 col=c("blue", "white", "red"),
                 show_row_names = T,
                 row_title="",
                 column_title="")
  
  myw <- 6
  myh <- 9
  
  hmpout <- sub("mat|txt|csv$","pdf", filein)
  
  cairo_pdf(hmpout, width=8, height=10)
  draw(hmp, heatmap_legend_side = "right", padding=unit(c(5,5,5,5), "mm")) # bottom, left, top, right
  dev.off()
  # system(sprintf("open %s", hmpout))
  
  csvout <- sub(".mat|.txt|.csv$","_query_words.csv", filein)
  neighbor_words_df <- do.call(rbind, neighbor_words)
  write.csv(neighbor_words_df, csvout, row.names = T, quote=F)
  system(sprintf("open %s", csvout))
}


















