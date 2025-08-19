# R code for proper reciprocals analysis plots
# Load required libraries
library(ggplot2)
library(ggrepel)  # For non-overlapping labels
library(readr)
library(dplyr)
library(RColorBrewer)
library(viridis)

# Load and prepare data
data <- read_csv("matrix_clean.csv")

# Create grouping variable
data <- data %>%
  mutate(
    group = case_when(
      lemma %in% c("each_other", "one_another") ~ "Reciprocal",
      lemma %in% c("someone", "anyone", "anything", "everything", "somebody", "anybody") ~ "Fused_Determinative",
      class == "pronoun" ~ "Pronoun", 
      class == "determinative" ~ "Other_Determinative",
      TRUE ~ "Other"
    ),
    # Create highlighting variable for key items
    highlight = lemma %in% c("each_other", "one_another", "someone", "anyone", "anything", "everything", "somebody", "anybody")
  )

# Prepare feature matrix for PCA
feature_matrix <- data %>% 
  select(-lemma, -class, -group, -highlight) %>%
  as.matrix()

# Perform PCA
pca_result <- prcomp(feature_matrix, scale. = TRUE)

# Create PCA dataframe
pca_data <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2], 
  lemma = data$lemma,
  group = data$group,
  highlight = data$highlight
)

# Calculate variance explained
var_explained <- summary(pca_result)$importance[2, 1:2] * 100

# Create the plot
p1 <- ggplot(pca_data, aes(x = PC1, y = PC2)) +
  # Plot all points first (smaller, muted)
  geom_point(aes(color = group, shape = group), 
             size = 1.5, alpha = 0.6) +
  
  # Overlay highlighted points (larger, more prominent)
  geom_point(data = filter(pca_data, highlight), 
             aes(color = group, shape = group),
             size = 3, alpha = 0.9, stroke = 1) +
  
  # Add repelled labels only for key items
  geom_text_repel(data = filter(pca_data, highlight),
                  aes(label = lemma, color = group),
                  size = 3.5,
                  fontface = "bold",
                  max.overlaps = Inf,
                  force = 2,
                  min.segment.length = 0.1,
                  segment.color = "grey50",
                  segment.size = 0.5,
                  box.padding = 0.5,
                  point.padding = 0.3) +
  
  # Custom colors - high contrast
  scale_color_manual(values = c(
    "Reciprocal" = "#FF6B35",           # Bright orange
    "Fused_Determinative" = "#004E89",   # Deep blue  
    "Pronoun" = "#1B9E77",              # Teal
    "Other_Determinative" = "#D95F02",   # Orange
    "Other" = "#7570B3"                 # Purple
  )) +
  
  # Custom shapes
  scale_shape_manual(values = c(
    "Reciprocal" = 17,                  # Triangle
    "Fused_Determinative" = 19,         # Circle
    "Pronoun" = 15,                     # Square
    "Other_Determinative" = 18,         # Diamond
    "Other" = 20                        # Small circle
  )) +
  
  # Labels and theme
  labs(
    x = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    title = "PCA Projection: Reciprocals Hypothesis Test",
    subtitle = "Reciprocals vs Fused Determinatives vs Pronouns",
    color = "Grammatical Category",
    shape = "Grammatical Category"
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    legend.title = element_text(size = 10, face = "bold"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, color = "grey20")
  ) +
  
  guides(
    color = guide_legend(override.aes = list(size = 4, alpha = 1)),
    shape = guide_legend(override.aes = list(size = 4, alpha = 1))
  )

# Save high-quality plots
ggsave("reciprocals_pca_projection.png", p1, 
       width = 12, height = 8, dpi = 300, bg = "white")
ggsave("reciprocals_pca_projection.pdf", p1, 
       width = 12, height = 8, bg = "white")

# Print the plot
print(p1)

# Create distance heatmap
key_items <- pca_data %>% 
  filter(highlight) %>%
  arrange(group, lemma)

# Fix indexing issue: get the row indices of highlighted items
highlight_indices <- which(data$highlight)
key_features <- feature_matrix[highlight_indices, ]
dist_matrix <- as.matrix(dist(key_features))
rownames(dist_matrix) <- colnames(dist_matrix) <- key_items$lemma

# Convert to long format for ggplot
library(reshape2)
dist_long <- melt(dist_matrix, varnames = c("Item1", "Item2"), value.name = "Distance")

# Create heatmap
p2 <- ggplot(dist_long, aes(x = Item1, y = Item2, fill = Distance)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = round(Distance, 2)), 
            color = "white", fontface = "bold", size = 3) +
  scale_fill_viridis_c(option = "plasma", direction = -1) +
  labs(
    title = "Distance Matrix: Key Linguistic Items",
    subtitle = "Euclidean distances in 155-dimensional feature space",
    x = "", y = "",
    fill = "Distance"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(hjust = 1),
    plot.title = element_text(size = 14, face = "bold"),
    panel.grid = element_blank()
  ) +
  coord_fixed()

# Save heatmap
ggsave("reciprocals_distance_heatmap.png", p2, 
       width = 10, height = 8, dpi = 300, bg = "white")
ggsave("reciprocals_distance_heatmap.pdf", p2, 
       width = 10, height = 8, bg = "white")

print(p2)

# Summary statistics
cat("PCA Summary:\n")
cat("PC1 explains", round(var_explained[1], 1), "% of variance\n")
cat("PC2 explains", round(var_explained[2], 1), "% of variance\n")
cat("Total variance explained by first 2 PCs:", round(sum(var_explained), 1), "%\n\n")

cat("Key items plotted:\n")
print(key_items %>% select(lemma, group))