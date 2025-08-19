# Generate distance heatmap for reciprocals analysis
# R code for publication-quality heatmap

library(ggplot2)
library(readr)
library(dplyr)
library(viridis)

# Load and prepare data
data <- read_csv("matrix_clean.csv")

# Select key items for heatmap
# Reciprocals + fused determinatives + representative pronouns
key_items <- c(
  # Reciprocals
  "each_other", "one_another",
  # Fused determinatives  
  "someone", "anyone", "anything", "everything", "somebody", "anybody",
  # Representative pronouns (selecting diverse types)
  "he", "she", "it", "they", "this", "that", "who", "what", "myself", "yourself"
)

# Filter data to key items
key_data <- data %>%
  filter(lemma %in% key_items) %>%
  arrange(match(lemma, key_items))  # Preserve order

# Create grouping for color coding
key_data <- key_data %>%
  mutate(
    group = case_when(
      lemma %in% c("each_other", "one_another") ~ "Reciprocal",
      lemma %in% c("someone", "anyone", "anything", "everything", "somebody", "anybody") ~ "Fused_Determinative",
      TRUE ~ "Pronoun"
    )
  )

# Extract feature matrix
feature_matrix <- key_data %>% 
  select(-lemma, -class, -group) %>%
  as.matrix()

# Calculate distance matrix (Euclidean distance in feature space)
dist_matrix <- as.matrix(dist(feature_matrix, method = "euclidean"))
rownames(dist_matrix) <- colnames(dist_matrix) <- key_data$lemma

# Convert to long format for ggplot using base R
dist_long <- expand.grid(Item1 = rownames(dist_matrix), 
                        Item2 = colnames(dist_matrix),
                        stringsAsFactors = FALSE)
dist_long$Distance <- as.vector(dist_matrix)

# Add grouping information for color coding
dist_long <- dist_long %>%
  left_join(key_data %>% select(lemma, group), by = c("Item1" = "lemma")) %>%
  rename(Group1 = group) %>%
  left_join(key_data %>% select(lemma, group), by = c("Item2" = "lemma")) %>%
  rename(Group2 = group)

# Create custom ordering for better visualization
# Order: Reciprocals, then Fused Determinatives, then Pronouns
item_order <- c(
  "each_other", "one_another",
  "someone", "anyone", "anything", "everything", "somebody", "anybody", 
  "he", "she", "it", "they", "this", "that", "who", "what", "myself", "yourself"
)

dist_long$Item1 <- factor(dist_long$Item1, levels = item_order)
dist_long$Item2 <- factor(dist_long$Item2, levels = rev(item_order))  # Reverse for matrix layout

# Create the heatmap
p_heatmap <- ggplot(dist_long, aes(x = Item1, y = Item2, fill = Distance)) +
  geom_tile(color = "white", linewidth = 0.2) +
  
  # Add text labels with distances
  geom_text(aes(label = round(Distance, 1)), 
            color = "white", size = 2.5, fontface = "bold") +
  
  # Use viridis color scale (good for accessibility and publication)
  scale_fill_viridis_c(option = "plasma", direction = 1, 
                       name = "Euclidean\nDistance") +
  
  # Labels and title
  labs(
    title = "Pairwise Distances Between Key Linguistic Items",
    subtitle = "Euclidean distances in 157-dimensional feature space",
    x = "",
    y = "",
    caption = "Lower values (darker colors) indicate greater similarity"
  ) +
  
  # Theme
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    plot.caption = element_text(size = 8, hjust = 0.5, face = "italic"),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  ) +
  
  # Equal aspect ratio for square tiles
  coord_fixed()

# Print summary statistics
cat("=== HEATMAP ANALYSIS ===\n")
cat("Items included:\n")
print(key_data %>% select(lemma, group))

cat("\nDistance summary statistics:\n")
cat("Range:", round(range(dist_matrix), 2), "\n")
cat("Mean:", round(mean(dist_matrix), 2), "\n")
cat("Median:", round(median(dist_matrix), 2), "\n")

# Calculate average distances between groups
cat("\nAverage distances between groups:\n")

# Reciprocals to Fused Determinatives
recip_to_fused <- dist_matrix[
  key_data$lemma[key_data$group == "Reciprocal"],
  key_data$lemma[key_data$group == "Fused_Determinative"]
]
cat("Reciprocals to Fused Determinatives:", round(mean(recip_to_fused), 2), "\n")

# Reciprocals to Pronouns  
recip_to_pronoun <- dist_matrix[
  key_data$lemma[key_data$group == "Reciprocal"],
  key_data$lemma[key_data$group == "Pronoun"]
]
cat("Reciprocals to Pronouns:", round(mean(recip_to_pronoun), 2), "\n")

# Difference
difference <- mean(recip_to_fused) - mean(recip_to_pronoun)
cat("Difference (Fused - Pronoun):", round(difference, 2))
if (difference > 0) {
  cat(" → Reciprocals closer to Pronouns\n")
} else {
  cat(" → Reciprocals closer to Fused Determinatives\n")
}

# Save high-quality plots
ggsave("reciprocals_distance_heatmap.png", p_heatmap, 
       width = 12, height = 10, dpi = 300, bg = "white")
ggsave("reciprocals_distance_heatmap.pdf", p_heatmap, 
       width = 12, height = 10, bg = "white")

# Display the plot
print(p_heatmap)

cat("\nFiles saved:\n")
cat("- reciprocals_distance_heatmap.png\n")
cat("- reciprocals_distance_heatmap.pdf\n")
cat("\nHeatmap analysis complete!\n")