# Standard linguistic analysis following established practices
# MCA + Distance-based ordination for categorical feature matrix

library(FactoMineR)
library(factoextra)
library(cluster)
library(vegan)
library(ggplot2)
library(dplyr)
library(readr)

# Load data
data <- read_csv("matrix_clean.csv")

# Prepare data for analysis
items <- data$lemma
categories <- data$class
feature_matrix <- data %>% select(-lemma, -class) %>% as.data.frame()
rownames(feature_matrix) <- items

# Create grouping for interpretation
groups <- case_when(
  items %in% c("each_other", "one_another") ~ "Reciprocal",
  items %in% c("someone", "anyone", "anything", "everything", "somebody", "anybody") ~ "Fused_Determinative",
  categories == "pronoun" ~ "Pronoun", 
  categories == "determinative" ~ "Other_Determinative",
  TRUE ~ "Other"
)

cat("=== STANDARD LINGUISTIC ANALYSIS ===\n")
cat("Following established practices in computational linguistics\n\n")

# 1. MULTIPLE CORRESPONDENCE ANALYSIS (MCA)
cat("1. Multiple Correspondence Analysis (MCA)\n")
cat("Standard approach for categorical linguistic features\n")
cat("----------------------------------------\n")

# Run MCA (treating binary features as categorical)
mca_result <- MCA(feature_matrix, graph = FALSE, ncp = 5)

# Extract variance explained
eigenvals <- get_eigenvalue(mca_result)
cat("Eigenvalues and variance explained:\n")
print(eigenvals[1:5, ])

# Create MCA biplot with contributions
p1 <- fviz_mca_ind(mca_result, 
                   axes = c(1, 2),
                   geom = "point",
                   habillage = groups,
                   addEllipses = TRUE,
                   ellipse.level = 0.68,
                   title = "MCA Biplot: Individual Items",
                   subtitle = paste0("Dim 1 (", round(eigenvals$variance.percent[1], 1), 
                                   "%) vs Dim 2 (", round(eigenvals$variance.percent[2], 1), "%)")) +
  theme_minimal() +
  scale_color_manual(values = c(
    "Reciprocal" = "#FF6B35",
    "Fused_Determinative" = "#004E89",
    "Pronoun" = "#1B9E77",
    "Other_Determinative" = "#D95F02",
    "Other" = "#7570B3"
  ))

# Save MCA plot
ggsave("mca_biplot.png", p1, width = 12, height = 8, dpi = 300, bg = "white")
ggsave("mca_biplot.pdf", p1, width = 12, height = 8, bg = "white")

# Variable contributions to first two dimensions
var_contrib <- get_mca_var(mca_result)
cat("\nTop contributing variables to Dimension 1:\n")
print(head(var_contrib$contrib[order(var_contrib$contrib[,1], decreasing = TRUE), 1], 10))

cat("\nTop contributing variables to Dimension 2:\n")
print(head(var_contrib$contrib[order(var_contrib$contrib[,2], decreasing = TRUE), 2], 10))

# 2. DISTANCE-BASED ORDINATION
cat("\n2. Distance-based Ordination\n")
cat("Standard dialectometry approach\n")
cat("--------------------------------\n")

# Calculate Jaccard distance (appropriate for binary sparse features)
# Jaccard ignores double zeros - standard for linguistic features
jaccard_dist <- vegdist(feature_matrix, method = "jaccard")

cat("Using Jaccard distance (standard for binary linguistic features)\n")
cat("Range of distances:", round(range(jaccard_dist), 3), "\n")

# Principal Coordinates Analysis (PCoA) - metric MDS
pcoa_result <- cmdscale(jaccard_dist, eig = TRUE, k = 2)

# Create PCoA dataframe
pcoa_data <- data.frame(
  PC1 = pcoa_result$points[, 1],
  PC2 = pcoa_result$points[, 2],
  Item = items,
  Group = groups
)

# Calculate variance explained by first two axes
eig_vals <- pcoa_result$eig[pcoa_result$eig > 0]
var_exp1 <- round(eig_vals[1] / sum(eig_vals) * 100, 1)
var_exp2 <- round(eig_vals[2] / sum(eig_vals) * 100, 1)

cat("PCoA variance explained:\n")
cat("PC1:", var_exp1, "%\n")
cat("PC2:", var_exp2, "%\n")

# Create PCoA plot
p2 <- ggplot(pcoa_data, aes(x = PC1, y = PC2, color = Group)) +
  geom_point(size = 2, alpha = 0.7) +
  stat_ellipse(level = 0.68, type = "norm") +
  scale_color_manual(values = c(
    "Reciprocal" = "#FF6B35",
    "Fused_Determinative" = "#004E89", 
    "Pronoun" = "#1B9E77",
    "Other_Determinative" = "#D95F02",
    "Other" = "#7570B3"
  )) +
  labs(
    x = paste0("PCo1 (", var_exp1, "% variance)"),
    y = paste0("PCo2 (", var_exp2, "% variance)"),
    title = "Principal Coordinates Analysis (PCoA)",
    subtitle = "Based on Jaccard distances - standard dialectometry approach",
    color = "Grammatical Category"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom"
  )

# Add labels for key items
key_items <- pcoa_data[pcoa_data$Group %in% c("Reciprocal", "Fused_Determinative"), ]
p2 <- p2 + 
  ggrepel::geom_text_repel(data = key_items, 
                          aes(label = Item), 
                          size = 3, 
                          max.overlaps = Inf,
                          box.padding = 0.5)

# Save PCoA plot  
ggsave("pcoa_jaccard.png", p2, width = 12, height = 8, dpi = 300, bg = "white")
ggsave("pcoa_jaccard.pdf", p2, width = 12, height = 8, bg = "white")

# 3. HIERARCHICAL CLUSTERING
cat("\n3. Hierarchical Clustering\n")
cat("Standard complement to ordination\n")
cat("---------------------------------\n")

# Ward clustering on Jaccard distances
hclust_result <- hclust(jaccard_dist, method = "ward.D2")

# Create dendrogram
png("dendrogram_ward.png", width = 1200, height = 800, res = 150)
plot(hclust_result, labels = items, main = "Ward Clustering on Jaccard Distances",
     sub = "Standard dialectometry approach", xlab = "Items", cex = 0.8)
dev.off()

# Cut tree to get clusters
k_clusters <- 5  # Reasonable number based on our groups
clusters <- cutree(hclust_result, k = k_clusters)

# Create cluster assignment table
cluster_table <- data.frame(
  Item = items,
  Theoretical_Group = groups,
  Cluster = clusters
)

cat("Cluster assignments for key items:\n")
key_cluster_table <- cluster_table[cluster_table$Theoretical_Group %in% 
                                  c("Reciprocal", "Fused_Determinative"), ]
print(key_cluster_table)

# 4. FOCUSED ANALYSIS: RECIPROCALS
cat("\n4. Focused Analysis: Reciprocal Items\n")
cat("------------------------------------\n")

# Extract coordinates for reciprocals
reciprocal_items <- c("each_other", "one_another")
recip_mca <- mca_result$ind$coord[items %in% reciprocal_items, 1:2]
recip_pcoa <- pcoa_data[pcoa_data$Item %in% reciprocal_items, c("PC1", "PC2")]

cat("MCA coordinates:\n")
print(data.frame(Item = reciprocal_items, recip_mca))

cat("\nPCoA coordinates:\n")  
print(data.frame(Item = reciprocal_items, recip_pcoa))

# Distance to group centroids in original space
fused_indices <- which(groups == "Fused_Determinative")
pronoun_indices <- which(groups == "Pronoun")

# Calculate centroids  
fused_centroid <- colMeans(feature_matrix[fused_indices, ])
pronoun_centroid <- colMeans(feature_matrix[pronoun_indices, ])

cat("\nDistances from reciprocals to group centroids (Jaccard):\n")
for (recip in reciprocal_items) {
  recip_index <- which(items == recip)
  recip_features <- feature_matrix[recip_index, ]
  
  # Calculate Jaccard distances to centroids
  dist_to_fused <- sum(pmin(recip_features, fused_centroid)) / 
                   sum(pmax(recip_features, fused_centroid))
  dist_to_pronoun <- sum(pmin(recip_features, pronoun_centroid)) / 
                     sum(pmax(recip_features, pronoun_centroid))
  
  cat(recip, ":\n")
  cat("  Distance to fused determinatives:", round(1 - dist_to_fused, 3), "\n")
  cat("  Distance to pronouns:", round(1 - dist_to_pronoun, 3), "\n")
  cat("  Closer to:", ifelse(dist_to_fused > dist_to_pronoun, "fused determinatives", "pronouns"), "\n\n")
}

cat("=== ANALYSIS COMPLETE ===\n")
cat("Files generated:\n")
cat("- mca_biplot.png/pdf (MCA analysis)\n")
cat("- pcoa_jaccard.png/pdf (Distance-based ordination)\n") 
cat("- dendrogram_ward.png (Hierarchical clustering)\n")
cat("\nThis follows established practices in computational linguistics\n")
cat("and provides interpretable, theory-grounded visualizations.\n")